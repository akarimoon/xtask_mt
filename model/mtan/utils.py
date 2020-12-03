import time
import torch
import torch.nn.functional as F
import numpy as np

"""
Define task metrics, loss functions and model trainer here.
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type, is_cs=False):
    device = x_pred.device
    ignore = 250 if is_cs else -1

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        if is_cs:
            loss = F.nll_loss(x_pred, x_output, ignore_index=ignore)
        else:
            loss = F.nll_loss(x_pred, x_output, ignore_index=ignore)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss


def compute_miou(x_pred, x_output, is_cs=False):
    ignore = 250 if is_cs else -1
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    class_nb = x_pred.size(1)
    device = x_pred.device
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        invalid_mask = (x_output[i] != ignore).float()
        for j in range(class_nb):
            pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
            true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
            mask_comb = pred_mask.float() + true_mask.float()
            union = torch.sum((mask_comb > 0).float() * invalid_mask)  # remove non-defined pixel predictions
            intsec = torch.sum((mask_comb > 1).float())
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size


def compute_iou(x_pred, x_output, is_cs=False):
    ignore = 250 if is_cs else -1
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        if i == 0:
            pixel_acc = torch.div(
                torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                torch.sum((x_output_label[i] != ignore).float()))
        else:
            pixel_acc = pixel_acc + torch.div(
                torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                torch.sum((x_output_label[i] != ignore).float()))
    return pixel_acc / batch_size


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


"""
=========== Universal Multi-task Trainer =========== 
"""


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, total_epoch=200, is_cs=False):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
    lambda_weight = np.ones([2, total_epoch])
    for index in range(total_epoch):
        cost = np.zeros(12, dtype=np.float32)
        start_time = time.time()

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        # iteration for all batches
        if is_cs:
            # iteration for all batches
            multi_task_model.train()
            train_dataset = iter(train_loader)
            for k in range(train_batch):
                train_data, train_label, train_depth = train_dataset.next()
                train_data, train_label = train_data.to(device), train_label.long().to(device)
                train_depth = train_depth.to(device)

                train_pred, logsigma = multi_task_model(train_data)

                optimizer.zero_grad()
                train_loss = [model_fit(train_pred[0], train_label, 'semantic', is_cs=True),
                            model_fit(train_pred[1], train_depth, 'depth', is_cs=True)]

                if opt.weight == 'equal' or opt.weight == 'dwa':
                    loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(2)])
                else:
                    loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(2))

                loss.backward()
                optimizer.step()

                cost[0] = train_loss[0].item()
                cost[1] = compute_miou(train_pred[0], train_label, is_cs=True).item()
                cost[2] = compute_iou(train_pred[0], train_label, is_cs=True).item()
                cost[3] = train_loss[1].item()
                cost[4], cost[5] = depth_error(train_pred[1], train_depth)
                avg_cost[index, :6] += cost[:6] / train_batch

            # evaluating test data
            multi_task_model.eval()
            with torch.no_grad():  # operations inside don't track history
                test_dataset = iter(test_loader)
                for k in range(test_batch):
                    test_data, test_label, test_depth = test_dataset.next()
                    test_data, test_label = test_data.to(device), test_label.long().to(device)
                    test_depth = test_depth.to(device)

                    test_pred, _ = multi_task_model(test_data)
                    test_loss = [model_fit(test_pred[0], test_label, 'semantic', is_cs=True),
                                model_fit(test_pred[1], test_depth, 'depth', is_cs=True)]

                    cost[6] = test_loss[0].item()
                    cost[7] = compute_miou(test_pred[0], test_label, is_cs=True).item()
                    cost[8] = compute_iou(test_pred[0], test_label, is_cs=True).item()
                    cost[9] = test_loss[1].item()
                    cost[10], cost[11] = depth_error(test_pred[1], test_depth)

                    avg_cost[index, 6:] += cost[6:] / test_batch

            scheduler.step()
            elapsed_time = (time.time() - start_time) / 60
            print("=======================================")
            print('Epoch: {:04d} [{:.1f}min]'.format(index, elapsed_time))
            print('Train Loss: segmt: {:.4f} --- depth: {:.4f}'.format(avg_cost[index, 0], avg_cost[index, 3]))
            print('Test Loss: segmt: {:.4f} --- depth: {:.4f}'.format(avg_cost[index, 6], avg_cost[index, 9]))
            print('Scores (Test):')
            print('Pix Acc: {:.4f}, mIoU: {:.4f} | Abs Err: {:.4f}, Rel Err: {:.4f}'.format(
                    avg_cost[index, 8], avg_cost[index, 7], 
                    avg_cost[index, 10], avg_cost[index, 11], 
            ))

        else:
            multi_task_model.train()
            train_dataset = iter(train_loader)
            for k in range(train_batch):
                train_data, train_label, train_depth, train_normal = train_dataset.next()
                train_data, train_label = train_data.to(device), train_label.long().to(device)
                train_depth, train_normal = train_depth.to(device), train_normal.to(device)

                train_pred, logsigma = multi_task_model(train_data)

                optimizer.zero_grad()
                train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                            model_fit(train_pred[1], train_depth, 'depth'),
                            model_fit(train_pred[2], train_normal, 'normal')]

                if opt.weight == 'equal' or opt.weight == 'dwa':
                    loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(3)])
                else:
                    loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

                loss.backward()
                optimizer.step()

                cost[0] = train_loss[0].item()
                cost[1] = compute_miou(train_pred[0], train_label).item()
                cost[2] = compute_iou(train_pred[0], train_label).item()
                cost[3] = train_loss[1].item()
                cost[4], cost[5] = depth_error(train_pred[1], train_depth)
                cost[6] = train_loss[2].item()
                cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
                avg_cost[index, :12] += cost[:12] / train_batch

            # evaluating test data
            multi_task_model.eval()
            with torch.no_grad():  # operations inside don't track history
                test_dataset = iter(test_loader)
                for k in range(test_batch):
                    test_data, test_label, test_depth, test_normal = test_dataset.next()
                    test_data, test_label = test_data.to(device), test_label.long().to(device)
                    test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                    test_pred, _ = multi_task_model(test_data)
                    test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                                model_fit(test_pred[1], test_depth, 'depth'),
                                model_fit(test_pred[2], test_normal, 'normal')]

                    cost[12] = test_loss[0].item()
                    cost[13] = compute_miou(test_pred[0], test_label).item()
                    cost[14] = compute_iou(test_pred[0], test_label).item()
                    cost[15] = test_loss[1].item()
                    cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                    cost[18] = test_loss[2].item()
                    cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)

                    avg_cost[index, 12:] += cost[12:] / test_batch

            scheduler.step()
            elapsed_time = (time.time() - start_time) / 60
            print("=======================================")
            print('Epoch: {:04d} [{:.1f}min]'.format(index, elapsed_time))
            print('Train Loss: segmt: {:.4f} --- depth: {:.4f} --- normal: {:.4f}'.format(avg_cost[index, 0], avg_cost[index, 3], avg_cost[index, 6]))
            print('Test Loss: segmt: {:.4f} --- depth: {:.4f} --- normal: {:.4f}'.format(avg_cost[index, 12], avg_cost[index, 15], avg_cost[index, 18]))
            print('Scores (Test):')
            print('Pix Acc: {:.4f}, mIoU: {:.4f} | Abs Err: {:.4f}, Rel Err: {:.4f} | mean: {:.4f}, med: {:.4f}, <11.25: {:.4f}, <22.5: {:.4f}, <30: {:.4f}'.format(
                    avg_cost[index, 14], avg_cost[index, 13], 
                    avg_cost[index, 16], avg_cost[index, 17], 
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]
            ))


"""
=========== Universal Single-task Trainer =========== 
"""


def single_task_trainer(train_loader, test_loader, single_task_model, device, optimizer, scheduler, opt, total_epoch=200):
    total_epoch = 200
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    for index in range(total_epoch):
        cost = np.zeros(24, dtype=np.float32)

        # iteration for all batches
        single_task_model.train()
        train_dataset = iter(train_loader)
        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred = single_task_model(train_data)
            optimizer.zero_grad()

            if opt.task == 'semantic':
                train_loss = model_fit(train_pred, train_label, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[0] = train_loss.item()
                cost[1] = compute_miou(train_pred, train_label).item()
                cost[2] = compute_iou(train_pred, train_label).item()

            if opt.task == 'depth':
                train_loss = model_fit(train_pred, train_depth, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[3] = train_loss.item()
                cost[4], cost[5] = depth_error(train_pred, train_depth)

            if opt.task == 'normal':
                train_loss = model_fit(train_pred, train_normal, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[6] = train_loss.item()
                cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

            avg_cost[index, :12] += cost[:12] / train_batch

        # evaluating test data
        single_task_model.eval()
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device),  test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = single_task_model(test_data)

                if opt.task == 'semantic':
                    test_loss = model_fit(test_pred, test_label, opt.task)
                    cost[12] = test_loss.item()
                    cost[13] = compute_miou(test_pred, test_label).item()
                    cost[14] = compute_iou(test_pred, test_label).item()

                if opt.task == 'depth':
                    test_loss = model_fit(test_pred, test_depth, opt.task)
                    cost[15] = test_loss.item()
                    cost[16], cost[17] = depth_error(test_pred, test_depth)

                if opt.task == 'normal':
                    test_loss = model_fit(test_pred, test_normal, opt.task)
                    cost[18] = test_loss.item()
                    cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred, test_normal)

                avg_cost[index, 12:] += cost[12:] / test_batch

        scheduler.step()
        if opt.task == 'semantic':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
        if opt.task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
        if opt.task == 'normal':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                      avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
