import copy
import random
import numpy as np
from math import exp
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Logger():
    """
    preds and targets are inverse depth
    """
    def __init__(self, num_classes=19, ignore_index=250):
        self.cm = 0
        self.mtan_miou = 0
        self.mtan_iou = 0
        self.rmse = 0
        self.irmse = 0
        self.irmse_log = 0
        self.abs = 0
        self.abs_rel = 0
        self.sqrt_rel = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.count = 0
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def _confusion_matrix(self, preds, targets, mask=None):
        preds = np.argmax(preds, axis=1)
        targets = np.squeeze(targets).astype(int)
        if mask is None:
            mask = np.ones_like(preds) == 1
        else:
            mask = np.squeeze(mask)
        k = (preds >= 0) & (preds < self.num_classes) & (preds != self.ignore_index)
        k &= (targets >= 0) & (targets < self.num_classes) & (targets != self.ignore_index)
        k &= (mask.astype(np.bool))
        return confusion_matrix(preds[k].flatten(), targets[k].flatten(), labels=list(range(self.num_classes)))
        # return np.bincount(self.num_classes * preds[k].astype(int) + targets[k], 
        #                     minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def _get_segmt_scores(self):
        if self.cm.sum() == 0:
            return 0, 0, 0
        pixel = np.diag(self.cm).sum() / self.cm.sum()
        perclass = np.diag(self.cm) / np.sum(self.cm, axis=1)
        iou = np.diag(self.cm) / (self.cm.sum(axis=1) + self.cm.sum(axis=0) - np.diag(self.cm))
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     pixel = np.diag(self.cm).sum() / np.float(self.cm.sum())
        #     perclass = np.diag(self.cm) / self.cm.sum(1).astype(np.float)
        #     IU = np.diag(self.cm) / (self.cm.sum(1) + self.cm.sum(0) - np.diag(self.cm)).astype(np.float)
        return pixel, np.nanmean(perclass), np.nanmean(iou)

    def _compute_miou(self, x_pred, x_output):
        """
        from https://github.com/lorenmt/mtan
        """
        x_pred = torch.from_numpy(x_pred)
        x_output = torch.from_numpy(x_output)
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        class_nb = x_pred.size(1)
        device = x_pred.device
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            invalid_mask = (x_output[i] >= 0).float()
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

    def _compute_iou(self, x_pred, x_output):
        """
        from https://github.com/lorenmt/mtan
        """
        x_pred = torch.from_numpy(x_pred)
        x_output = torch.from_numpy(x_output)
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(
                    torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                    torch.sum((x_output_label[i] >= 0).float()))
            else:
                pixel_acc = pixel_acc + torch.div(
                    torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                    torch.sum((x_output_label[i] >= 0).float()))
        return pixel_acc / batch_size

    def _depth_rmse(self, preds, targets, masks):
        return np.sum(np.sqrt(np.abs(preds - targets) ** 2 * masks)) / np.sum(masks)

    def _depth_irmse(self, inv_preds, inv_targets, masks):
        return np.sum(np.sqrt(np.abs(inv_preds - inv_targets) ** 2 * masks)) / np.sum(masks)

    def _depth_irmse_log(self, inv_preds, inv_targets, masks):
        return np.sum(np.sqrt(np.abs(np.log(inv_preds) - np.log(inv_targets)) ** 2 * masks)) / np.sum(masks)

    def _depth_abs(self, preds, targets, masks):
        nonzero = targets > 0
        absdiff = np.abs(preds[nonzero] - targets[nonzero])
        return np.sum(absdiff) / np.sum(nonzero)
    
    def _depth_abs_rel(self, preds, targets, masks):
        nonzero = targets > 0
        absdiff = np.abs(preds[nonzero] - targets[nonzero])
        relabsdiff = absdiff / targets[nonzero]
        return np.sum(relabsdiff) / np.sum(nonzero)

    def _depth_sqrt_rel(self, preds, targets, masks):
        nonzero = targets > 0
        sqrtdiff = np.abs(preds[nonzero] - targets[nonzero]) ** 2
        relsqrtdiff = sqrtdiff / targets[nonzero]
        return np.sum(relsqrtdiff) / np.sum(nonzero)

    def _depth_acc(self, preds, targets, masks, thres):
        preds = np.clip(preds, a_min=1e-9, a_max=None)
        targets = np.clip(targets, a_min=1e-9, a_max=None)
        maxratio = np.fmax(preds / targets, targets / preds) * masks
        maxratio[masks == 0.] += 1e5
        counts = np.sum((maxratio < thres).astype(float), axis=(2, 3))
        return np.mean(counts / np.sum(masks, axis=(2, 3)))

    def log(self, preds, targets, masks=None):
        """
        preds are inverse depth
        """
        preds = [p.cpu().numpy() for p in preds]
        targets = [t.cpu().numpy() for t in targets]
        masks = [m.cpu().numpy() for m in masks]
        preds_segmt, preds_depth = preds
        targets_segmt, targets_depth = targets
        masks_segmt, masks_depth = masks

        inv_preds_depth = np.copy(preds_depth)
        inv_targets_depth = np.copy(targets_depth)
        preds_depth = 1 / preds_depth
        targets_depth[targets_depth > 0] = 1 / targets_depth[targets_depth > 0]

        N = preds_segmt.shape[0]
        self.cm += self._confusion_matrix(preds_segmt, targets_segmt, masks_segmt)
        self.mtan_miou += self._compute_miou(preds_segmt, targets_segmt) * N
        self.mtan_iou += self._compute_iou(preds_segmt, targets_segmt) * N
        self.rmse += self._depth_rmse(preds_depth, targets_depth, masks_depth) * N
        self.irmse += self._depth_irmse(inv_preds_depth, inv_targets_depth, masks_depth) * N
        # self.irmse_log += self._depth_irmse_log(inv_preds, inv_targets, masks_depth) * N
        self.abs += self._depth_abs(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.abs_rel += self._depth_abs_rel(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.sqrt_rel += self._depth_sqrt_rel(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.delta1 += self._depth_acc(preds_depth, targets_depth, masks_depth, thres=1.25) * N
        self.delta2 += self._depth_acc(preds_depth, targets_depth, masks_depth, thres=1.25**2) * N
        self.delta3 += self._depth_acc(preds_depth, targets_depth, masks_depth, thres=1.25**3) * N
        self.count += N

    def get_scores(self):
        self.pixel_acc, self.mean_acc, self.iou = self._get_segmt_scores()
        self.mtan_miou /= self.count
        self.mtan_iou /= self.count
        self.rmse /= self.count
        self.irmse /= self.count
        # self.irmse_log /= self.count
        self.abs /= self.count
        self.abs_rel /= self.count
        self.sqrt_rel /= self.count
        self.delta1 /= self.count
        self.delta2 /= self.count
        self.delta3 /= self.count

        print_segmt_str = "Pix Acc: {:.3f}, Mean acc: {:.3f}, IoU: {:.3f}"
        print(print_segmt_str.format(
            self.pixel_acc, self.mean_acc, self.iou
        ))

        print_segmt_mtan_str = "(from MTAN): mIoU: {:.3f}, IoU: {:.3f}"
        print(print_segmt_mtan_str.format(
            self.mtan_miou, self.mtan_iou
        ))

        print_depth_str = "Scores - RMSE: {:.4f}, iRMSE: {:.4f}, Abs: {:.4f}, Abs Rel: {:.4f}, " +\
            "delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}"
        print(print_depth_str.format(
            self.rmse, self.irmse, self.abs, self.abs_rel, self.delta1, self.delta2, self.delta3
        ))

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class MaskedKLLoss(nn.Module):
    def __init__(self, n_classes, label_smoothing):
        """
        KL Divergence Loss with mask
        - predicted: (N, C, H, W) -> (N, H * W, C)
        - target: (N, 1, H, W) -> (N, H * W, 1)
        - mask: (N, 1, H, W) -> (N, H * W, 1)
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing
        self.n_classes = n_classes
        smoothing_value = self.label_smoothing / (self.n_classes - 2)
        self.one_hot = torch.full((self.n_classes, ), smoothing_value)
        self.one_hot = self.one_hot.unsqueeze(0)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, predicted, target, mask=None):
        N, C, H, W = predicted.shape
        predicted = predicted.view(N, H * W, C)
        target = target.view(N, H * W, 1)
        if mask is None:
            mask = torch.ones(target.shape)
        else:
            mask = mask.view(N, H * W, 1)
        self.one_hot = self.one_hot.to(predicted.device)

        model_prob = self.one_hot.repeat(N, H * W, 1)
        model_prob = model_prob.scatter_(2, target, value=self.confidence)
        model_prob_masked = model_prob * mask
        predicted = predicted * mask

        return self.loss_fn(F.log_softmax(predicted, dim=2), model_prob_masked)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, label_smoothing=0.0, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - label_smoothing
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, predicted, target, mask=None):
        with torch.no_grad():
            # true_dist = predicted.data.clone()
            true_dist = torch.zeros_like(predicted)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if mask is not None:
            predicted *= mask
            true_dist *= mask
        return torch.mean(torch.sum(-true_dist * F.log_softmax(predicted, dim=self.dim), dim=self.dim))

class XTaskLoss(nn.Module):
    def __init__(self, num_classes=19, alpha=0.2, gamma=0., label_smoothing=0.,
                 image_loss_type="MSE", t_segmt_loss_type="cross", grad_loss=False):
        super(XTaskLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.image_loss_type = image_loss_type
        self.grad_loss = grad_loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=250, reduction='mean')

        if t_segmt_loss_type == "kl":
            self.kl_loss = MaskedKLLoss(n_classes=num_classes, label_smoothing=label_smoothing)
        elif t_segmt_loss_type == "cross":
            self.kl_loss = LabelSmoothingLoss(num_classes=num_classes, label_smoothing=label_smoothing)

    def masked_SSIM(self, predicted, target, mask):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        predicted *= mask
        target *= mask

        mu_pred = nn.AvgPool2d(3, 1)(predicted)
        mu_targ = nn.AvgPool2d(3, 1)(target)
        mu_pred_mu_targ = mu_pred * mu_targ
        mu_pred_sq = mu_pred.pow(2)
        mu_targ_sq = mu_targ.pow(2)

        sigma_pred = nn.AvgPool2d(3, 1)(predicted * predicted) - mu_pred_sq
        sigma_targ = nn.AvgPool2d(3, 1)(target * target) - mu_targ_sq
        sigma_predtarg = nn.AvgPool2d(3, 1)(predicted * target) - mu_pred_mu_targ

        SSIM_n = (2 * mu_pred_mu_targ + C1) * (2 * sigma_predtarg + C2)
        SSIM_d = (mu_pred_sq + mu_targ_sq + C1) * (sigma_pred + sigma_targ + C2)
        SSIM = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

        return torch.mean(torch.sum(SSIM, dim=(2,3)) / torch.sum(mask, dim=(2,3)))

    def masked_L1_loss(self, predicted, target, mask):
        diff = torch.abs(predicted - target) * mask
        loss = torch.sum(diff, dim=(2,3)) / torch.sum(mask, dim=(2,3))
        return torch.mean(loss)

    def masked_mse_loss(self, predicted, target, mask):
        diff = torch.pow((predicted - target), 2) * mask
        loss = torch.sum(diff, dim=(2,3)) / torch.sum(mask, dim=(2,3))
        return torch.mean(loss)

    def masked_logL1_loss(self, predicted, target, mask):
        diff = torch.log(1 + torch.abs(predicted - target)) * mask
        loss = torch.sum(diff, dim=(2,3)) / torch.sum(mask, dim=(2,3))
        return torch.mean(loss)

    def _calc_gradient(self, x, scale=True):
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        dx, dy = right - left, bottom - top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def depth_grad_loss(self, predicted, target, mask):
        dx, dy = self._calc_gradient(predicted - target, scale=True)
        dx *= mask
        dy *= mask
        loss = torch.sum(torch.abs(dx) + torch.abs(dy), dim=(2, 3)) / torch.sum(mask, dim=(2, 3))
        return torch.mean(loss)


    def forward(self, predicted, targ_segmt, targ_depth, mask_segmt=None, mask_depth=None, log_vars=None):
        pred_segmt, pred_t_segmt, pred_depth, pred_t_depth = predicted
        if mask_segmt is None:
            mask_segmt = torch.ones_like(targ_segmt)
        if mask_depth is None:
            mask_depth = torch.ones_like(targ_depth)

        if self.grad_loss:
            grad_loss = self.depth_grad_loss(pred_depth, targ_depth, mask_depth)
        else:
            grad_loss = 0

        if self.image_loss_type =="L1":
            depth_loss = self.masked_L1_loss(pred_depth, targ_depth, mask_depth)
        elif self.image_loss_type == "MSE":
            depth_loss = self.masked_mse_loss(pred_depth, targ_depth, mask_depth)
        elif self.image_loss_type == "logL1":
            depth_loss = self.masked_logL1_loss(pred_depth, targ_depth, mask_depth)
        ssim_loss = self.masked_SSIM(pred_depth.clone(), pred_t_depth.clone(), mask_depth)

        segmt_loss = self.cross_entropy_loss(pred_segmt, targ_segmt)
        kl_loss = self.kl_loss(pred_t_segmt.clone(), torch.argmax(pred_segmt.clone(), dim=1), mask_segmt)
        
        if log_vars is None:
            image_loss = (1 - self.alpha) * depth_loss + self.alpha * ssim_loss
            label_loss = (1 - self.gamma) * segmt_loss + self.gamma * kl_loss

        else:
            # image_loss = torch.exp(-log_vars[0]) * depth_loss + torch.exp(-log_vars[1]) * ssim_loss + \
            #              log_vars[0] + log_vars[1]
            # label_loss = torch.exp(-log_vars[2]) * segmt_loss + torch.exp(-log_vars[3]) * kl_loss + \
            #              log_vars[2] + log_vars[3]
            image_loss_tmp = (1 - self.alpha) * depth_loss + self.alpha * ssim_loss + grad_loss
            label_loss_tmp = (1 - self.gamma) * segmt_loss + self.gamma * kl_loss
            image_loss = 0.5 * torch.exp(-log_vars[0]) * image_loss_tmp + log_vars[0]
            label_loss = torch.exp(-log_vars[1]) * label_loss_tmp + log_vars[1]

        return image_loss, label_loss

class PCGrad():
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        input:
        - objectives: a list of objectives
        '''

        grads, shapes = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, shapes=None):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        pc_grad = torch.stack(pc_grad).mean(dim=0)
        return pc_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes = [], []
        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
        return grads, shapes

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network.
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        '''

        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
        return grad, shape
