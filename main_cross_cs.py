import argparse, sys, os, time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import CityscapesDataset
from parser import cityscapes_xtask_parser
from module import Logger, XTaskLoss
from model.xtask_ts import XTaskTSNet
from utils import *

DEPTH_CORRECTION = 2.1116e-09

def compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                 batch_mask_segmt, batch_mask_depth, 
                 model, log_vars=None,
                 criterion=None, optimizer=None, is_train=True):

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)
    batch_mask_segmt = batch_mask_segmt.to(device, non_blocking=True)
    batch_mask_depth = batch_mask_depth.to(device, non_blocking=True)

    output = model(batch_X)
    image_loss, label_loss = criterion(output, batch_y_segmt, batch_y_depth,
                                       batch_mask_segmt, batch_mask_depth, log_vars=log_vars)

    if is_train:
        optimizer.zero_grad()
        image_loss.backward(retain_graph=True)
        label_loss.backward()
        optimizer.step()

    return image_loss.item() + label_loss.item()

if __name__=='__main__':
    torch.manual_seed(0)
    opt = cityscapes_xtask_parser()
    opt.betas = (opt.b1, opt.b2)

    print("Initializing...")
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    if not opt.debug:
        if not opt.infer_only:
            exp_num, results_dir = make_results_dir()
        else:
            exp_num = str(opt.exp_num).zfill(3)
            results_dir = os.path.join(opt.save_path, exp_num)
        weights_path = os.path.join(results_dir, "model", "model.pth")
    else:
        exp_num = ""
        results_dir = opt.save_path

    parameters_to_train = []

    print("Parameters:")
    print("   predicting at size [{}*{}]".format(opt.height, opt.width))
    if opt.num_classes != 19:
        print("   # of classes: {}".format(opt.num_classes))
    print("   using ResNet{}, optimizer: Adam (lr={}, beta={}), scheduler: StepLR({}, {})".format(
        opt.enc_layers, opt.lr, opt.betas, opt.scheduler_step_size, opt.scheduler_gamma))
    print("   loss function --- Lp_depth: {}, tsegmt: {}, alpha: {}, gamma: {}, smoothing: {}".format(
        opt.lp, opt.tseg_loss, opt.alpha, opt.gamma, opt.label_smoothing))
    print("   batch size: {}, train for {} epochs".format(
        opt.batch_size, opt.epochs))

    device_name = "cpu" if opt.cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=opt.enc_layers, out_features_segmt=opt.num_classes)
    model.to(device)
    parameters_to_train = [p for p in model.parameters()]
    print("TransferNet type:")
    print("   {}".format(model.trans_name))
    
    print("Options:")
    log_vars = None
    if opt.uncertainty_weights:
        print("   use uncertainty weights")
        """
        Implementation of uncertainty weights (learnable weight parameters to balance losses of multiple tasks)
        See arxiv.org/abs/1705.07115
        """
        log_var_a = torch.zeros((1,), requires_grad=True, device=device_name)
        log_var_b = torch.zeros((1,), requires_grad=True, device=device_name)
        log_vars = [log_var_a, log_var_b]
        parameters_to_train += log_vars
    if opt.grad_loss:
        print("   use grad loss (k=1), no scaling")

    print("Loading dataset...")
    train_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                   split='train', transform=["random_flip"])
    valid_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                   split='val', transform=None)
    # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    valid = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    criterion = XTaskLoss(num_classes=opt.num_classes, alpha=opt.alpha, gamma=opt.gamma, label_smoothing=opt.label_smoothing,
                          image_loss_type=opt.lp, t_segmt_loss_type=opt.tseg_loss, grad_loss=opt.grad_loss).to(device)
    optimizer = optim.Adam(parameters_to_train, lr=opt.lr, betas=opt.betas)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.scheduler_step_size, opt.scheduler_gamma)

    if not opt.infer_only:
        print("=======================================")
        print("Start training...")
        best_valid_loss = 1e5
        train_losses = []
        valid_losses = []
        save_at_epoch = 0

        for epoch in range(1, opt.epochs + 1):

            start = time.time()
            train_loss = 0.
            valid_loss = 0.

            for i, batch in enumerate(tqdm(train, disable=opt.notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                    batch_mask_segmt, batch_mask_depth, 
                                    model, log_vars=log_vars,
                                    criterion=criterion, optimizer=optimizer, is_train=True)
                train_loss += loss

            for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                    batch_mask_segmt, batch_mask_depth, 
                                    model, log_vars=log_vars,
                                    criterion=criterion, optimizer=optimizer, is_train=False)
                valid_loss += loss

            train_loss /= len(train.dataset)
            valid_loss /= len(valid.dataset)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            elapsed_time = (time.time() - start) / 60
            print("Epoch {}/{} [{:.1f}min] --- train loss: {:.5f} --- valid loss: {:.5f}".format(
                        epoch, opt.epochs, elapsed_time, train_loss, valid_loss))
            if opt.uncertainty_weights:
                print("Uncertainty weights: segmt={:.5f}, depth={:.5f}".format(
                        (torch.exp(log_vars[1]) ** 0.5).item(), (torch.exp(log_vars[0]) ** 0.5).item()))

            if not opt.debug:
                if epoch == 0 or valid_loss < best_valid_loss:
                    print("Saving weights...")
                    weights = model.state_dict()
                    torch.save(weights, weights_path)
                    best_valid_loss = valid_loss
                    save_at_epoch = epoch

            scheduler.step()

        print("Training done")
        print("=======================================")

        train_losses = np.array(train_losses)
        valid_losses = np.array(valid_losses)

        np.save(os.path.join(results_dir, "model", "tr_losses.npy".format(opt.alpha, opt.gamma)), train_losses)
        np.save(os.path.join(results_dir, "model", "va_losses.npy".format(opt.alpha, opt.gamma)), valid_losses)

    else:
        print("Infer only mode -> skip training...")

    if not opt.debug:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    logger = Logger(num_classes=opt.num_classes)
    best_loss = 1e5

    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
            original, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
            batch_y_depth = batch_y_depth.to(device, non_blocking=True)
            batch_mask_segmt = batch_mask_segmt.to(device, non_blocking=True)
            batch_mask_depth = batch_mask_depth.to(device, non_blocking=True)

            predicted = model(batch_X)
            image_loss, label_loss = criterion(predicted, batch_y_segmt, batch_y_depth,
                                               batch_mask_segmt, batch_mask_depth)

            pred_segmt, pred_t_segmt, pred_depth, pred_t_depth = predicted

            preds = [pred_segmt, pred_depth]
            targets = [batch_y_segmt, batch_y_depth]
            masks = [batch_mask_segmt, batch_mask_depth]
            logger.log(preds, targets, masks)

            # use best results for final plot
            loss = compute_miou(pred_segmt, batch_y_segmt) - depth_error(pred_depth, batch_y_depth)[0]
            if i == 0 or loss < best_loss:
                best_loss = loss
                best_original = original
                best_y_depth = batch_y_depth
                best_pred_segmt = pred_segmt
                best_pred_depth = pred_depth
                best_pred_tsegmt = pred_t_segmt
                best_pred_tdepth = pred_t_depth
            
    logger.get_scores()

    if not opt.infer_only:
        write_results(logger, opt, model, exp_num=exp_num)
        write_indv_results(opt, model, folder_path=results_dir)

    show = 0
    if not opt.infer_only:
        plt.figure(figsize=(14, 8))
        plt.plot(np.arange(opt.epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(opt.epochs), valid_losses, linestyle="--", label="valid")
        plt.legend()
        if not opt.view_only:
            plt.savefig(os.path.join(results_dir, "output", "loss.png".format(opt.batch_size, opt.alpha, opt.gamma)))

    plt.figure(figsize=(18, 10))
    plt.subplot(3,3,1)
    plt.imshow(best_original[0][show].cpu().numpy())
    plt.title("Image")

    if not opt.infer_only:
        plt.subplot(3,3,2)
        plt.plot(np.arange(opt.epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(opt.epochs), valid_losses, linestyle="--", label="valid")
        plt.title("Loss")
        plt.legend()
        
    plt.subplot(3,3,4)
    plt.imshow(valid_data.decode_segmt(torch.argmax(best_pred_segmt, dim=1)[show].cpu().numpy()))
    plt.title("Direct segmt. pred.")

    plt.subplot(3,3,5)
    plt.imshow(valid_data.decode_segmt(torch.argmax(best_pred_tsegmt, dim=1)[show].cpu().numpy()))
    plt.title("Cross-task segmt. pred.")

    plt.subplot(3,3,6)
    plt.imshow(valid_data.decode_segmt(best_original[1][show].cpu().numpy()))
    plt.title("Segmt. target")

    plt.subplot(3,3,7)
    pred_clamped = torch.clamp(best_pred_depth, min=1e-9, max=1.)
    plt.imshow(pred_clamped[show].squeeze().cpu().numpy())
    plt.title("Direct depth pred.")

    plt.subplot(3,3,8)
    pred_t_clamped = torch.clamp(best_pred_tdepth, min=1e-9, max=1.)
    plt.imshow(pred_t_clamped[show].squeeze().cpu().numpy())
    plt.title("Cross-task depth pred.")

    plt.subplot(3,3,9)
    plt.imshow(best_original[2][show].squeeze().cpu().numpy())
    plt.title("Depth target")

    plt.tight_layout()
    ep_or_infer = "epoch{}-{}_".format(save_at_epoch, opt.epochs) if not opt.infer_only else "infer_"
    if not opt.view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "results.png".format(opt.batch_size, opt.alpha, opt.gamma)))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    img = ax1.imshow(np.abs((1 / best_pred_depth[show] - 1 / best_y_depth[show]).squeeze().cpu().numpy()))
    fig.colorbar(img, ax=ax1)
    plt.title("Absolute error of depth (non-inverted)")

    flat_pred = torch.flatten(best_pred_depth[show]).cpu().numpy()
    flat_targ = torch.flatten(best_y_depth[show]).cpu().numpy()
    # sns.histplot(1 / flat_pred[flat_pred > 0], stat='density', color='blue', label='pred', ax=ax2)
    # sns.histplot(1 / flat_targ[flat_targ > 0], stat='density', color='green', label='target', ax=ax2)
    # plt.title("Density plot of depth (non-inverted")
    # plt.legend()

    df = pd.DataFrame()
    df["pred"] = 1 / flat_pred[flat_targ > 0]
    df["targ"] = 1 / flat_targ[flat_targ > 0]
    df["diff_abs"] = np.abs(df["pred"] - df["targ"])
    bins = np.linspace(0, 500, 51)
    df["targ_bin"] = np.digitize(np.round(df["targ"]), bins) - 1
    sns.boxplot(x="targ_bin", y="diff_abs", data=df, ax=ax3)
    ax3.set_xticklabels([int(t.get_text()) * 10  for t in ax3.get_xticklabels()])
    ax3.set_title("Boxplot for absolute error for all non-nan pixels")

    df["is_below_20"] = df["targ"] < 20
    bins_20 = np.linspace(0, 20, 21)
    df["targ_bin_20"] = np.digitize(np.round(df["targ"]), bins_20) - 1
    sns.boxplot(x="targ_bin_20", y="diff_abs", data=df[df["is_below_20"] == True], ax=ax4)
    ax4.set_xticklabels([int(t.get_text()) * 1  for t in ax4.get_xticklabels()])
    ax4.set_title("Boxplot for absolute error for all pixels < 20m")
    plt.tight_layout()
    if not opt.view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "hist.png".format(opt.batch_size, opt.alpha, opt.gamma)))
    
    if not opt.run_only:
        plt.show()