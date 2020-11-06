import argparse, sys, os, time
import cv2
import numpy as np
import pandas as pd
import torch
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import CityscapesDataset
from parser import cityscapes_xtask_parser
from module import Logger, XTaskLoss, PCGrad
from model.xtask_ts import XTaskTSNet
from utils import *

DEPTH_CORRECTION = 2.1116e-09

def compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                 batch_mask_segmt, batch_mask_depth, 
                 model, log_vars=None,
                 criterion=None, optimizer=None, is_train=True, use_pcgrad=False):

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
        if use_pcgrad:
            optimizer.pc_backward([image_loss, label_loss])
        else:
            image_loss.backward(retain_graph=True)
            label_loss.backward()
        optimizer.step()

    return image_loss.item() + label_loss.item()

if __name__=='__main__':
    torch.manual_seed(0)
    args = cityscapes_xtask_parser()

    print("Initializing...")
    input_path = args.input_path

    enc_layers = args.enc_layers
    height = args.height
    width = args.width

    lr = args.lr
    beta_1 = args.b1
    beta_2 = args.b2
    betas = (beta_1, beta_2)
    alpha = args.alpha
    gamma = args.gamma
    label_smoothing = args.label_smoothing
    lp = args.lp
    t_segmt = args.tseg_loss

    use_uncertainty = args.uncertainty_weights
    use_pcgrad = args.pcgrad
    use_gradloss = args.grad_loss

    batch_size = args.batch_size
    num_epochs = args.epochs
    num_classes = args.num_classes
 
    num_workers = args.workers

    infer_only = args.infer_only
    view_only = args.view_only
    use_cpu = args.cpu
    debug_mode = args.debug
    notqdm = args.notqdm

    if not debug_mode:
        if not infer_only:
            exp_num, results_dir = make_results_dir()
        else:
            exp_num = str(args.exp_num).zfill(3)
            results_dir = os.path.join("./tmp", exp_num)
        weights_path = os.path.join(results_dir, "model", "model.pth")
    else:
        exp_num = ""
        results_dir = "./tmp"

    parameters_to_train = []

    print("Parameters:")
    print("   predicting at size [{}*{}]".format(height, width))
    if num_classes != 19:
        print("   # of classes: {}".format(num_classes))
    print("   using ResNet{}, optimizer: Adam (lr={}, beta={}), scheduler: StepLR(15, 0.1)".format(enc_layers, lr, betas))
    print("   loss function --- Lp_depth: {}, tsegmt: {}, alpha: {}, gamma: {}, smoothing: {}".format(lp, t_segmt, alpha, gamma, label_smoothing))
    print("   batch size: {}, train for {} epochs".format(batch_size, num_epochs))

    device_name = "cpu" if use_cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=enc_layers, out_features_segmt=num_classes)
    model.to(device)
    parameters_to_train = [p for p in model.parameters()]
    print("TransferNet type:")
    print("   {}".format(model.trans_name))
    
    print("Options:")
    log_vars = None
    if use_uncertainty:
        print("   use uncertainty weights")
        log_var_a = torch.zeros((1,), requires_grad=True, device=device_name)
        log_var_b = torch.zeros((1,), requires_grad=True, device=device_name)
        # log_var_c = torch.zeros((1,), requires_grad=True, device=device_name)
        # log_var_d = torch.zeros((1,), requires_grad=True, device=device_name)
        log_vars = [log_var_a, log_var_b]
        parameters_to_train += log_vars
    if use_pcgrad:
        print("   use pcgrad")    
    if use_gradloss:
        print("   use grad loss (k=1), no scaling")

    print("Loading dataset...")
    train_data = CityscapesDataset(root_path=input_path, height=height, width=width, num_classes=num_classes,
                                   split='train', transform=["random_flip"])
    valid_data = CityscapesDataset(root_path=input_path, height=height, width=width, num_classes=num_classes,
                                   split='val', transform=None)
    # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = XTaskLoss(num_classes=num_classes, alpha=alpha, gamma=gamma, label_smoothing=label_smoothing,
                          image_loss_type=lp, t_segmt_loss_type=t_segmt, grad_loss=use_gradloss).to(device)
    if use_pcgrad:
        optimizer = PCGrad(optim.Adam(parameters_to_train, lr=lr, betas=betas))
    else:
        optimizer = optim.Adam(parameters_to_train, lr=lr, betas=betas)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1)

    if not infer_only:
        print("=======================================")
        print("Start training...")
        best_valid_loss = 1e5
        train_losses = []
        valid_losses = []
        save_at_epoch = 0

        for epoch in range(1, num_epochs + 1):

            start = time.time()
            train_loss = 0.
            valid_loss = 0.

            for i, batch in enumerate(tqdm(train, disable=notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                    batch_mask_segmt, batch_mask_depth, 
                                    model, log_vars=log_vars,
                                    criterion=criterion, optimizer=optimizer, is_train=True, use_pcgrad=use_pcgrad)
                train_loss += loss

            for i, batch in enumerate(tqdm(valid, disable=notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                    batch_mask_segmt, batch_mask_depth, 
                                    model, log_vars=log_vars,
                                    criterion=criterion, optimizer=optimizer, is_train=False, use_pcgrad=use_pcgrad)
                valid_loss += loss

            train_loss /= len(train.dataset)
            valid_loss /= len(valid.dataset)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            elapsed_time = (time.time() - start) / 60
            print("Epoch {}/{} [{:.1f}min] --- train loss: {:.5f} --- valid loss: {:.5f}".format(
                        epoch, num_epochs, elapsed_time, train_loss, valid_loss))
            if use_uncertainty:
                print("Uncertainty weights: segmt={:.5f}, depth={:.5f}".format(
                        (torch.exp(log_vars[1]) ** 0.5).item(), (torch.exp(log_vars[0]) ** 0.5).item()))

            if not debug_mode:
                if epoch == 0 or valid_loss < best_valid_loss:
                    print("Saving weights...")
                    weights = model.state_dict()
                    torch.save(weights, weights_path)
                    best_valid_loss = valid_loss
                    save_at_epoch = epoch

            if not use_pcgrad:
                scheduler.step()

        print("Training done")
        print("=======================================")

        train_losses = np.array(train_losses)
        valid_losses = np.array(valid_losses)

        np.save(os.path.join(results_dir, "model", "tr_losses.npy".format(alpha, gamma)), train_losses)
        np.save(os.path.join(results_dir, "model", "va_losses.npy".format(alpha, gamma)), valid_losses)

    else:
        print("Infer only mode -> skip training...")

    if not debug_mode:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    logger = Logger(num_classes=num_classes)
    best_loss = 1e5

    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid, disable=notqdm)):
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

            loss = image_loss.item() + label_loss.item()
            if i == 0 or loss < best_loss:
                best_loss = loss
                best_original = original
                best_y_depth = batch_y_depth
                best_pred_segmt = pred_segmt
                best_pred_depth = pred_depth
                best_pred_tsegmt = pred_t_segmt
                best_pred_tdepth = pred_t_depth
            
    logger.get_scores()

    if not infer_only:
        write_results(logger, args, model, exp_num=exp_num)
        write_indv_results(args, model, folder_path=results_dir)

    show = 2
    if not infer_only:
        plt.figure(figsize=(14, 8))
        plt.plot(np.arange(num_epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(num_epochs), valid_losses, linestyle="--", label="valid")
        plt.legend()
        if not view_only:
            plt.savefig(os.path.join(results_dir, "output", "loss.png".format(batch_size, alpha, gamma)))

    plt.figure(figsize=(18, 10))
    plt.subplot(3,3,1)
    plt.imshow(best_original[0][show].cpu().numpy())
    plt.title("Image")

    if not infer_only:
        plt.subplot(3,3,2)
        plt.plot(np.arange(num_epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(num_epochs), valid_losses, linestyle="--", label="valid")
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
    ep_or_infer = "epoch{}-{}_".format(save_at_epoch, num_epochs) if not infer_only else "infer_"
    if not view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "results.png".format(batch_size, alpha, gamma)))

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
    if not view_only:
        plt.savefig(os.path.join(results_dir, "output", ep_or_infer + "hist.png".format(batch_size, alpha, gamma)))
    
    plt.show()