import numpy as np
import torch

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