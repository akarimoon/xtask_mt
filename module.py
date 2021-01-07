import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class STLLogger():
    """
    preds and targets are inverse depth
    """
    def __init__(self, task, num_classes=7, ignore_index=250):
        self.pixel_acc = 0
        self.iou = []
        self.miou = 0
        self.abs = 0
        self.abs_rel = 0
        self.count = 0
        self.task = task
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def _compute_miou(self, x_pred, x_output):
        """
        from https://github.com/lorenmt/mtan
        """
        x_pred = torch.from_numpy(x_pred)
        x_output = torch.from_numpy(x_output)
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        device = x_pred.device

        batch_avg = 0
        ious = []
        for i in range(batch_size):
            true_class = 0
            class_prob = 0
            mask = (x_output[i] != self.ignore_index).float()
            batch_ious = []

            for j in range(self.num_classes):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
                mask_comb = pred_mask.float() + true_mask.float()
                union = torch.sum((mask_comb > 0).float() * mask)  # remove non-defined pixel predictions
                intsec = torch.sum((mask_comb > 1).float())
                if union == 0:
                    batch_ious.append(np.nan)
                    continue
                class_prob += intsec / union
                true_class += 1
                batch_ious.append((intsec / union).item())
                
            batch_avg += class_prob / true_class
            ious.append(batch_ious)
        
        ious = np.array(ious)
        self.iou.append(np.nanmean(ious, axis=0))

        return batch_avg / batch_size

    def _compute_pixacc(self, x_pred, x_output):
        """
        from https://github.com/lorenmt/mtan
        """
        x_pred = torch.from_numpy(x_pred)
        x_output = torch.from_numpy(x_output)
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        pixel_acc = 0
        for i in range(batch_size):
            pixel_acc += torch.div(
                            torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                            torch.sum((x_output_label[i] != self.ignore_index).float())
                        )
        return pixel_acc / batch_size

    def _depth_abs(self, preds, targets, masks):
        """
        Calculate absolute error of inverse depth
        """
        nonzero = targets > 0
        absdiff = np.abs(preds[nonzero] - targets[nonzero])
        return np.sum(absdiff) / np.sum(nonzero)
    
    def _depth_abs_rel(self, preds, targets, masks):
        """
        Calculate absolute relative error of inverse depth
        """
        nonzero = targets > 0
        absdiff = np.abs(preds[nonzero] - targets[nonzero])
        relabsdiff = absdiff / targets[nonzero]
        return np.sum(relabsdiff) / np.sum(nonzero)

    def log(self, pred, target, mask=None):
        """
        preds are inverse depth
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        if mask is not None:
            mask = mask.cpu().numpy()
        else:
            mask = np.ones_like(target)
        N = pred.shape[0]

        if self.task == 'segmt':
            self.pixel_acc += self._compute_pixacc(pred, target) * N
            self.miou += self._compute_miou(pred, target) * N

        elif self.task == 'depth':
            inv_pred = np.copy(pred)
            inv_target = np.copy(target)
            self.abs += self._depth_abs(inv_pred, inv_target, mask) * N
            self.abs_rel += self._depth_abs_rel(inv_pred, inv_target, mask) * N
        self.count += N

    def get_scores(self):
        self.pixel_acc /= self.count
        self.miou /= self.count
        self.abs /= self.count
        self.abs_rel /= self.count

        if self.task == 'segmt':
            print_segmt_str = "Pix Acc: {:.4f}, mIoU: {:.4f}, IoU: {}"
            print(print_segmt_str.format(
                self.pixel_acc, self.miou, np.around(np.mean(self.iou, axis=0), decimals=4)
            ))

        elif self.task == 'depth':
            print_depth_str = "Scores - Abs: {:.4f}, Abs Rel: {:.4f}"
            print(print_depth_str.format(
                self.abs, self.abs_rel
            ))

class Logger():
    """
    preds and targets are inverse depth
    """
    def __init__(self, num_classes=19, ignore_index=250):
        self.pixel_acc = 0
        self.iou = []
        self.miou = 0
        self.rmse = 0
        self.irmse = 0
        self.irmse_log = 0
        self.iproj = 0
        self.abs = 0
        self.abs_rel = 0
        self.sqrt_rel = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.count = 0
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def _compute_miou(self, x_pred, x_output):
        """
        from https://github.com/lorenmt/mtan
        """
        x_pred = torch.from_numpy(x_pred)
        x_output = torch.from_numpy(x_output)
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        device = x_pred.device

        batch_avg = 0
        ious = []
        for i in range(batch_size):
            true_class = 0
            class_prob = 0
            mask = (x_output[i] != self.ignore_index).float()
            batch_ious = []

            for j in range(self.num_classes):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
                mask_comb = pred_mask.float() + true_mask.float()
                union = torch.sum((mask_comb > 0).float() * mask)  # remove non-defined pixel predictions
                intsec = torch.sum((mask_comb > 1).float())
                if union == 0:
                    batch_ious.append(np.nan)
                    continue
                class_prob += intsec / union
                true_class += 1
                batch_ious.append((intsec / union).item())
                
            batch_avg += class_prob / true_class
            ious.append(batch_ious)
        
        ious = np.array(ious)
        self.iou.append(np.nanmean(ious, axis=0))

        return batch_avg / batch_size

    def _compute_pixacc(self, x_pred, x_output):
        """
        from https://github.com/lorenmt/mtan
        """
        x_pred = torch.from_numpy(x_pred)
        x_output = torch.from_numpy(x_output)
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        pixel_acc = 0
        for i in range(batch_size):
            pixel_acc += torch.div(
                            torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
                            torch.sum((x_output_label[i] != self.ignore_index).float())
                        )
        return pixel_acc / batch_size

    def _depth_rmse(self, preds, targets, masks):
        """
        Calculate RMSE of actual depth
        """
        return np.sum(np.sqrt(np.abs(preds - targets) ** 2 * masks)) / np.sum(masks)

    def _depth_iproj_error(self, inv_preds, inv_targets, masks):
        inv_preds = np.clip(inv_preds, a_min=1e-9, a_max=None)
        flat_preds = (np.log(inv_preds) * masks).flatten()
        flat_projs = flat_preds * np.ones_like(flat_preds) / np.sqrt(len(flat_preds))
        return np.sum(np.dot(flat_preds, flat_preds) - np.dot(flat_projs, flat_projs)) / np.sum(masks)

    def _depth_irmse(self, inv_preds, inv_targets, masks):
        """
        Calculate RMSE of inverse depth
        """
        return np.sum(np.sqrt(np.abs(inv_preds - inv_targets) ** 2 * masks)) / np.sum(masks)

    def _depth_irmse_log(self, inv_preds, inv_targets, masks):
        """
        Calculate log RMSE of inverse depth
        """
        inv_preds = np.clip(inv_preds, a_min=1e-9, a_max=None)
        inv_targets = np.clip(inv_targets, a_min=1e-9, a_max=None)
        return np.sum(np.sqrt(np.abs(np.log(inv_preds) - np.log(inv_targets)) ** 2 * masks)) / np.sum(masks)

    def _depth_abs(self, preds, targets, masks):
        """
        Calculate absolute error of inverse depth
        """
        nonzero = targets > 0
        absdiff = np.abs(preds[nonzero] - targets[nonzero])
        return np.sum(absdiff) / np.sum(nonzero)
    
    def _depth_abs_rel(self, preds, targets, masks):
        """
        Calculate absolute relative error of inverse depth
        """
        nonzero = targets > 0
        absdiff = np.abs(preds[nonzero] - targets[nonzero])
        relabsdiff = absdiff / targets[nonzero]
        return np.sum(relabsdiff) / np.sum(nonzero)

    def _depth_sqrt_rel(self, preds, targets, masks):
        """
        Calculate square relative error of inverse depth
        """
        nonzero = targets > 0
        sqrtdiff = np.abs(preds[nonzero] - targets[nonzero]) ** 2
        relsqrtdiff = sqrtdiff / targets[nonzero]
        return np.sum(relsqrtdiff) / np.sum(nonzero)

    def _depth_acc(self, preds, targets, masks, thres):
        """
        Calculate accuracy of depth
        """
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
        preds_segmt, preds_depth = preds
        targets_segmt, targets_depth = targets
        if masks is not None:
            masks = [m.cpu().numpy() for m in masks]
            masks_segmt, masks_depth = masks
        else:
            masks_segmt = np.ones_like(targets_segmt)
            masks_depth = np.ones_like(targets_depth)

        inv_preds_depth = np.copy(preds_depth)
        inv_targets_depth = np.copy(targets_depth)
        preds_depth = 1 / preds_depth
        targets_depth[targets_depth > 0] = 1 / targets_depth[targets_depth > 0]

        N = preds_segmt.shape[0]
        self.pixel_acc += self._compute_pixacc(preds_segmt, targets_segmt) * N
        self.miou += self._compute_miou(preds_segmt, targets_segmt) * N
        self.rmse += self._depth_rmse(preds_depth, targets_depth, masks_depth) * N
        self.irmse += self._depth_irmse(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.irmse_log += self._depth_irmse_log(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.iproj += self._depth_iproj_error(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.abs += self._depth_abs(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.abs_rel += self._depth_abs_rel(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.sqrt_rel += self._depth_sqrt_rel(inv_preds_depth, inv_targets_depth, masks_depth) * N
        self.delta1 += self._depth_acc(preds_depth, targets_depth, masks_depth, thres=1.25) * N
        self.delta2 += self._depth_acc(preds_depth, targets_depth, masks_depth, thres=1.25**2) * N
        self.delta3 += self._depth_acc(preds_depth, targets_depth, masks_depth, thres=1.25**3) * N
        self.count += N

    def get_scores(self):
        self.pixel_acc /= self.count
        self.miou /= self.count
        self.rmse /= self.count
        self.irmse /= self.count
        self.irmse_log /= self.count
        self.iproj /= self.count
        self.abs /= self.count
        self.abs_rel /= self.count
        self.sqrt_rel /= self.count
        self.delta1 /= self.count
        self.delta2 /= self.count
        self.delta3 /= self.count

        print_segmt_str = "Pix Acc: {:.4f}, mIoU: {:.4f}, IoU: {}"
        print(print_segmt_str.format(
            self.pixel_acc, self.miou, np.around(np.mean(self.iou, axis=0), decimals=4)
        ))

        print_depth_str = "Scores - RMSE: {:.4f}, iRMSE: {:.4f}, iRMSE log: {:.4f}, iProjE: {:.4f}, Abs: {:.4f}, Abs Rel: {:.4f}, Sqrt Rel: {:.4f}, " +\
            "delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}"
        print(print_depth_str.format(
            self.rmse, self.irmse, self.irmse_log, self.iproj, self.abs, self.abs_rel, self.sqrt_rel, self.delta1, self.delta2, self.delta3
        ))



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
    """
    Cross Entropy loss with mask and smoothing
    """
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
    def __init__(self, num_classes=19, alpha=0.01, gamma=0.01, label_smoothing=0.,
                 image_loss_type="L1", t_segmt_loss_type="cross", t_depth_loss_type="L1",
                 balance_method=None, 
                 ignore_index=250):
        super(XTaskLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.image_loss_type = image_loss_type
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        self.t_depth_loss_type = t_depth_loss_type
        self.balance_method = balance_method

        self.nonlinear = nn.LogSoftmax(dim=1)

        if image_loss_type == "L1":
            self.image_loss = self.masked_L1_loss
        elif image_loss_type == "MSE":
            self.image_loss = self.masked_mse_loss
        elif image_loss_type == "logL1":
            self.image_loss = self.masked_logL1_loss
        elif image_loss_type == "smoothL1":
            self.image_loss = self.masked_smoothL1_loss

        if t_depth_loss_type == "ssim":
            self.tdep_loss = self.masked_SSIM
        elif t_depth_loss_type == "L1":
            self.tdep_loss = self.masked_L1_loss

        if t_segmt_loss_type == "kl":
            self.tseg_loss = MaskedKLLoss(n_classes=num_classes, label_smoothing=label_smoothing)
        elif t_segmt_loss_type == "cross":
            self.tseg_loss = LabelSmoothingLoss(num_classes=num_classes, label_smoothing=label_smoothing)

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

    def masked_smoothL1_loss(self, predicted, target, mask, beta=1.0):
        absdiff = torch.abs(predicted - target)
        diff = torch.where(absdiff < beta, 0.5 * absdiff ** 2 / beta, absdiff - 0.5 * beta) * mask
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

    def forward(self, predicted, targ_segmt, targ_depth, mask_segmt=None, mask_depth=None, task_weights=None, use_xtc=False):
        pred_segmt, pred_t_segmt, pred_depth, pred_t_depth = predicted
        if self.num_classes != 13 and mask_segmt is None:
            mask_segmt = torch.ones_like(targ_segmt)
        if mask_depth is None:
            mask_depth = torch.ones_like(targ_depth)

        depth_loss = self.image_loss(pred_depth, targ_depth, mask_depth)
        if not use_xtc:
            tdep_loss = self.tdep_loss(pred_t_depth, pred_depth.detach().clone(), mask_depth)
        else:
            tdep_loss = self.image_loss(pred_t_depth, targ_depth.clone(), mask_depth)

        segmt_loss = self.cross_entropy_loss(pred_segmt, targ_segmt)
        if not use_xtc:
            tseg_loss = self.tseg_loss(pred_t_segmt, torch.argmax(self.nonlinear(pred_segmt.detach().clone()), dim=1), mask_segmt)
        else:
            tseg_loss = self.cross_entropy_loss(pred_t_segmt, targ_segmt.clone())

        if self.balance_method is None:
            image_loss = (1 - self.alpha) * depth_loss + self.alpha * tdep_loss
            label_loss = (1 - self.gamma) * segmt_loss + self.gamma * tseg_loss

        elif self.balance_method == "uncert":
            # image_loss_tmp = (1 - self.alpha) * depth_loss + self.alpha * tdep_loss
            # label_loss_tmp = (1 - self.gamma) * segmt_loss + self.gamma * tseg_loss
            image_loss_tmp = (1 - self.alpha) * depth_loss + self.alpha * tseg_loss
            label_loss_tmp = (1 - self.gamma) * segmt_loss + self.gamma * tdep_loss
            image_loss = 0.5 * torch.exp(-task_weights[0]) * image_loss_tmp + task_weights[0]
            label_loss = 0.5 * torch.exp(-task_weights[1]) * label_loss_tmp + task_weights[1]

        elif self.balance_method == "gradnorm":
            image_loss = (1 - self.alpha) * depth_loss + self.alpha * tseg_loss
            label_loss = (1 - self.gamma) * segmt_loss + self.gamma * tdep_loss

        return image_loss, label_loss