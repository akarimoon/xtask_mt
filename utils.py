import os
from datetime import datetime
import torch

def write_results(logger, opt, model, file_path="./tmp/results.txt", exp_num=None):
    with open(file_path, 'a') as f:
        f.write("=" * 10 + "\n")
        if exp_num is not None:
            f.write("Experiment #{}\n".format(exp_num))
        f.write("Parameters: enc={}, lr={}, beta={}, lp={}, tsegmt={}, tdepth={}, alpha={}, gamma={}, temp:{}, smoothing={}\n".format(
            opt.enc_layers, opt.lr, (opt.b1, opt.b2), opt.lp, opt.tseg_loss, opt.tdep_loss, opt.alpha, opt.gamma, opt.temp, opt.label_smoothing
        ))
        f.write("Optimizer: Adam, Scheduler: StepLR(step size={}, gamma={})\n".format(opt.scheduler_step_size, opt.scheduler_gamma))
        if opt.num_classes != 19:
            f.write("# of classes: {}\n".format(opt.num_classes))
        f.write("shallow decoder: {} (if not written, then True)\n".format(opt.is_shallow))
        f.write("transfernet type: {}, use_uncertainty: {}, use gradloss: {}\n".format(
            model.trans_name, opt.uncertainty_weights, opt.grad_loss))
            
        print_segmt_str = "Pix Acc: {:.4f}, mIoU: {:.4f}\n"
        f.write(print_segmt_str.format(
            logger.pixel_acc, logger.miou
        ))
        print_depth_str = "Scores - RMSE: {:.4f}, iRMSE: {:.4f}, iRMSE log: {:.4f}, Abs: {:.4f}, Abs Rel: {:.4f}, Sqrt Rel: {:.4f}, " +\
            "delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}\n"
        f.write(print_depth_str.format(
            logger.rmse, logger.irmse, logger.irmse_log, logger.abs, logger.abs_rel, logger.sqrt_rel,
            logger.delta1, logger.delta2, logger.delta3
        ))

def write_indv_results(opt, model, folder_path):
    with open(os.path.join(folder_path, "indv_results.txt"), "a") as f:
        now = datetime.now()
        f.write("Date: {}\n".format(now.strftime("%b-%d-%Y %H:%M:%S")))
        f.write("arguments:\n")
        f.write("   predicting at size [{}*{}]\n".format(opt.height, opt.width))
        f.write("   batch size: {}, train for {} epochs\n".format(opt.batch_size, opt.epochs))
        f.write("   optimizer: Adam, scheduler: StepLR(step size={}, gamma={})\n".format(opt.scheduler_step_size, opt.scheduler_gamma))
        f.write("   enc={}, numclasses={}, lr={}, beta={}, lp={}, tsegmt={}, tdepth={}, alpha={}, gamma={}, smoothing={}\n".format(
            opt.enc_layers, opt.num_classes, opt.lr, (opt.b1, opt.b2), opt.lp, opt.tseg_loss, opt.tdep_loss,
            opt.alpha, opt.gamma, opt.label_smoothing
        ))
        f.write("   shallow decoder: {} (if not written, then True)\n".format(opt.is_shallow))
        f.write("   transfernet type: {}, use_uncertainty: {}, use gradloss: {}\n".format(
            model.trans_name, opt.uncertainty_weights, opt.grad_loss))

def make_results_dir(folder_path="./tmp"):
    i = 1
    while True:
        num = str(i).zfill(3)
        if not os.path.exists(os.path.join(folder_path, num)):
            os.mkdir(os.path.join(folder_path, num))
            break
        else:
            i += 1

    os.mkdir(os.path.join(folder_path, num, "model"))
    os.mkdir(os.path.join(folder_path, num, "output"))

    return num, os.path.join(folder_path, num)

"""
From MTAN
https://github.com/lorenmt/mtan
"""
def compute_miou(x_pred, x_output, ignore_index=250):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    class_nb = x_pred.size(1)
    device = x_pred.device
    batch_avg = 0
    for i in range(batch_size):
        true_class = 0
        class_prob = 0
        invalid_mask = (x_output[i] != ignore_index).float()
        for j in range(class_nb):
            pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
            true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
            mask_comb = pred_mask.float() + true_mask.float()
            union = torch.sum((mask_comb > 0).float() * invalid_mask)  # remove non-defined pixel predictions
            intsec = torch.sum((mask_comb > 1).float())
            if union == 0:
                continue
            class_prob += intsec / union
            true_class += 1
        
        batch_avg += class_prob / true_class
    return batch_avg / batch_size


def compute_iou(x_pred, x_output, ignore_index=250):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    pixel_acc = 0
    for i in range(batch_size):
        pixel_acc += torch.div(
                        torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                        torch.sum((x_output_label[i] != ignore_index).float())
                        )
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