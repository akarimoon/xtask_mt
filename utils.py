import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

def write_results(logger, opt, model, file_path="./exps/results.txt", exp_num=None):
    with open(file_path, 'a') as f:
        f.write("=" * 10 + "\n")
        if exp_num is not None:
            f.write("Experiment #{}\n".format(exp_num))
        f.write("Parameters: enc={}, lr={}, beta={}, lp={}, tsegmt={}, tdepth={}, alpha={}, gamma={}, smoothing={}\n".format(
            opt.enc_layers, opt.lr, (opt.b1, opt.b2), opt.lp, opt.tseg_loss, opt.tdep_loss, opt.alpha, opt.gamma, opt.label_smoothing
        ))
        f.write("Use pretrained encoder: {} (if not written, then False)\n".format(opt.use_pretrain))
        f.write("Optimizer: Adam, Scheduler: StepLR(step size={}, gamma={})\n".format(opt.scheduler_step_size, opt.scheduler_gamma))
        if opt.num_classes != 19:
            f.write("# of classes: {}\n".format(opt.num_classes))
        f.write("   batch norm in ttnet: {} (if not written, then False)\n".format(opt.batch_norm))
        f.write("   wider ttnet: {} (if not written, then False)\n".format(opt.wider_ttnet))
        f.write("transfernet type: {}, use_uncertainty: {}, use_gradnorm: {}\n".format(
            model.trans_name, opt.uncertainty_weights, opt.gradnorm))
            
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
        f.write("   use pretrained encoder: {} (if not written, then False)\n".format(opt.use_pretrain))
        f.write("   batch norm in ttnet: {} (if not written, then False)\n".format(opt.batch_norm))
        f.write("   wider ttnet: {} (if not written, then False)\n".format(opt.wider_ttnet))
        f.write("   transfernet type: {}, use_uncertainty: {}, use_gradnorm: {}\n".format(
            model.trans_name, opt.uncertainty_weights, opt.gradnorm))

def make_results_dir(folder_path="./exps"):
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

def make_plots(opt, results_dir, best_set, save_at_epoch, valid_data, train_losses=None, valid_losses=None, is_nyu=False):
    # best_set_size = best_set["pred_depth"].shape[0]
    best_set_size = min(opt.batch_size, 4)
    show = np.random.randint(best_set_size)
    if train_losses is not None and valid_losses is not None :
        plt.figure(figsize=(14, 8))
        plt.plot(np.arange(opt.epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(opt.epochs), valid_losses, linestyle="--", label="valid")
        plt.legend()
        if not opt.view_only:
            plt.savefig(os.path.join(results_dir, "output", "loss.png"))

    plt.figure(figsize=(18, 10))
    plt.subplot(3,3,1)
    if not is_nyu:
        plt.imshow(best_set["original"][0][show].cpu().numpy())
    else:
        plt.imshow(best_set["original"][show].permute(1,2,0).cpu().numpy())
    plt.title("Image")

    if train_losses is not None and valid_losses is not None :
        plt.subplot(3,3,2)
        plt.plot(np.arange(opt.epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(opt.epochs), valid_losses, linestyle="--", label="valid")
        plt.title("Loss")
        plt.legend()
        
    plt.subplot(3,3,4)
    plt.imshow(valid_data.decode_segmt(torch.argmax(best_set["pred_segmt"], dim=1)[show].cpu().numpy()))
    plt.title("Direct segmt. pred.")

    if best_set["pred_tsegmt"] is not None:
        plt.subplot(3,3,5)
        plt.imshow(valid_data.decode_segmt(torch.argmax(best_set["pred_tsegmt"], dim=1)[show].cpu().numpy()))
        plt.title("Cross-task segmt. pred.")

    plt.subplot(3,3,6)
    plt.imshow(valid_data.decode_segmt(best_set["targ_segmt"][show].cpu().numpy()))
    plt.title("Segmt. target")

    plt.subplot(3,3,7)
    pred_clamped = torch.clamp(best_set["pred_depth"], min=1e-9, max=0.4922) if not is_nyu else best_set["pred_depth"]
    plt.imshow(pred_clamped[show].squeeze().cpu().numpy())
    plt.title("Direct depth pred.")

    if best_set["pred_tdepth"] is not None:
        plt.subplot(3,3,8)
        pred_t_clamped = torch.clamp(best_set["pred_tdepth"], min=1e-9, max=0.4922) if not is_nyu else best_set["pred_tdepth"]
        plt.imshow(pred_t_clamped[show].squeeze().cpu().numpy())
        plt.title("Cross-task depth pred.")

    plt.subplot(3,3,9)
    plt.imshow(best_set["targ_depth"][show].squeeze().cpu().numpy())
    plt.title("Depth target")

    plt.tight_layout()
    ep_or_infer = "epoch{}-{}_".format(save_at_epoch, opt.epochs) if not opt.infer_only else "infer_"
    if not opt.view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "results.png"))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    img = ax1.imshow(np.abs((pred_clamped[show] - best_set["targ_depth"][show]).squeeze().cpu().numpy()))
    fig.colorbar(img, ax=ax1)
    plt.title("Absolute error of depth (non-inverted)")

    flat_pred = torch.flatten(best_set["pred_depth"][show]).cpu().numpy()
    flat_targ = torch.flatten(best_set["targ_depth"][show]).cpu().numpy()

    df = pd.DataFrame()
    if not is_nyu:
        df["pred"] = (0.20 * 2262) / (256 * flat_pred[flat_targ > 0])
        df["targ"] = (0.20 * 2262) / (256 * flat_targ[flat_targ > 0])
    else:
        df["pred"] = flat_pred[flat_targ > 0]
        df["targ"] = flat_targ[flat_targ > 0]
    df["diff_abs"] = np.abs(df["pred"] - df["targ"])
    bins = np.linspace(0, 500, 51)
    df["targ_bin"] = np.digitize(np.round(df["targ"]), bins) - 1

    sns.regplot(x="targ", y="pred", data=df, ax=ax2, scatter_kws={'s':5})
    ax2.set_xlim((df["targ"].min(), df["targ"].max()))
    ax2.set_ylim((df["targ"].min(), df["targ"].max()))
    # sns.regplot(x=flat_pred[flat_targ > 0], y=flat_targ[flat_targ > 0], ax=ax2)
    # ax2.set_xlim((0, 0.5))
    # ax2.set_ylim((0, 0.5))
    plt.title("Scatter plot of depth (non-inverted)")

    sns.boxplot(x="targ_bin", y="diff_abs", data=df, ax=ax3)
    ax3.set_xticklabels([int(t.get_text()) * 10  for t in ax3.get_xticklabels()])
    ax3.set_title("Boxplot for absolute error for all non-nan pixels")

    thres = 20 if not is_nyu else 10
    df["is_below_thres"] = df["targ"] < thres
    bins_thres = np.linspace(0, thres, thres + 1)
    df["targ_bin_thres"] = np.digitize(np.round(df["targ"]), bins_thres) - 1
    sns.boxplot(x="targ_bin_thres", y="diff_abs", data=df[df["is_below_thres"] == True], ax=ax4)
    ax4.set_xticklabels([int(t.get_text()) * 1  for t in ax4.get_xticklabels()])
    ax4.set_title("Boxplot for absolute error for all pixels < {}m".format(thres))
    plt.tight_layout()
    if not opt.view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "hist.png"))

    plt.figure(figsize=(4 * best_set_size, 12))
    for i in range(best_set_size):
        plt.subplot(5, best_set_size, i + 1)
        if not is_nyu:
            plt.imshow(best_set["original"][0][i].cpu().numpy())
        else:
            plt.imshow(best_set["original"][i].permute(1,2,0).cpu().numpy())
        plt.axis('off')
        plt.subplot(5, best_set_size, i + 1 + best_set_size)
        plt.imshow(valid_data.decode_segmt(best_set["targ_segmt"][i].cpu().numpy()))
        plt.axis('off')
        plt.subplot(5, best_set_size, i + 1 + 2 * best_set_size)
        plt.imshow(valid_data.decode_segmt(torch.argmax(best_set["pred_segmt"], dim=1)[i].cpu().numpy()))
        plt.axis('off')
        plt.subplot(5, best_set_size, i + 1 + 3 * best_set_size)
        plt.imshow(best_set["targ_depth"][i].squeeze().cpu().numpy())
        plt.axis('off')
        plt.subplot(5, best_set_size, i + 1 + 4 * best_set_size)
        plt.imshow(pred_clamped[i].squeeze().cpu().numpy())
        plt.axis('off')
    plt.tight_layout()
    if not opt.view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "batch.png"), bbox_inches='tight')

    if not opt.run_only:
        plt.show()

"""
From MTAN
https://github.com/lorenmt/mtan
"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def overall_score(x_preds, x_outputs, ignore_index=250):
    miou = compute_miou(x_preds[0], x_outputs[0])
    iou = compute_iou(x_preds[0], x_outputs[0])
    abs_err = depth_error(x_preds[1], x_outputs[1])[0]
    rel_err = depth_error(x_preds[1], x_outputs[1])[1]
    return miou - abs_err