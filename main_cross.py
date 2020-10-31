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
from module import Logger, XTaskLoss
from model.xtask_ts import XTaskTSNet

DEPTH_CORRECTION = 2.1116e-09

def compute_loss(batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth, model,
                 criterion=None, optimizer=None, is_train=True):

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)
    batch_mask_segmt = batch_mask_segmt.to(device, non_blocking=True)
    batch_mask_depth = batch_mask_depth.to(device, non_blocking=True)

    output = model(batch_X)
    image_loss, label_loss = criterion(output, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth)

    if is_train:
        optimizer.zero_grad()
        image_loss.backward(retain_graph=True)
        with torch.autograd.set_detect_anomaly(True):
            label_loss.backward()
        optimizer.step()

    return image_loss.item() + label_loss.item()

def write_results(logger, args, file_path="./tmp/output/grid_search_results.txt"):
    with open(file_path, 'a') as f:
        f.write("=" * 10 + "\n")
        f.write("Parameters: enc={}, lr={}, beta={}, lp={}, alpha={}, gamma={}, smoothing={}\n".format(
            args.enc_layers, args.lr, (args.b1, args.b2), args.lp, args.alpha, args.gamma, args.label_smoothing
        ))
        print_segmt_str = "Pix Acc: {:.3f}, Mean acc: {:.3f}, IoU: {:.3f}\n"
        f.write(print_segmt_str.format(
            logger.glob, logger.mean, logger.iou
        ))

        print_depth_str = "Scores - RMSE: {:.4f}, iRMSE: {:.4f}, Abs Rel: {:.4f}, Sqrt Rel: {:.4f}, " +\
            "delta1: {:.4f}, delta2: {:.4f}, delta3: {:.4f}\n"
        f.write(print_depth_str.format(
            logger.rmse, logger.irmse, logger.abs_rel, logger.sqrt_rel, logger.delta1, logger.delta2, logger.delta3
        ))

if __name__=='__main__':
    torch.manual_seed(0)
    args = cityscapes_xtask_parser()

    print("Initializing...")
    input_path = args.input_path
    # weights_path = args.save_weights

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

    batch_size = args.batch_size
    num_epochs = args.epochs
 
    num_workers = args.workers

    infer_only = args.infer_only
    use_cpu = args.cpu
    debug_mode = args.debug

    weights_path = "./tmp/model/xtask_alpha{}_gamma{}.pth".format(alpha, gamma)

    print("Parameters:")
    print("   predicting at size [{}*{}]".format(height, width))
    print("   using ResNet{}, optimizer: Adam (lr={}, beta={}), scheduler: StepLR(15, 0.1)".format(enc_layers, lr, betas))
    print("   loss function --- Lp_depth: " + lp + ", alpha: {}, gamma: {}, smoothing: {}".format(alpha, gamma, label_smoothing))
    print("   batch size: {}, train for {} epochs".format(batch_size, num_epochs))

    device_name = "cpu" if use_cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=enc_layers)
    model.to(device)

    print("Loading dataset...")
    train_data = CityscapesDataset(root_path=input_path, height=height, width=width,
                                   split='train', transform=["random_flip"])
    valid_data = CityscapesDataset(root_path=input_path, height=height, width=width, 
                                   split='val', transform=None)
    # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    criterion = XTaskLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing, image_loss_type=lp).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
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

            for i, batch in enumerate(tqdm(train)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth, model,
                                    criterion=criterion, optimizer=optimizer, is_train=True)
                train_loss += loss

            for i, batch in enumerate(tqdm(valid)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth, model,
                                    criterion=criterion, optimizer=optimizer, is_train=False)
                valid_loss += loss

            train_loss /= len(train.dataset)
            valid_loss /= len(valid.dataset)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            elapsed_time = (time.time() - start) / 60
            print("Epoch {}/{} [{:.1f}min] --- train loss: {:.5f} --- valid loss: {:.5f}".format(
                        epoch, num_epochs, elapsed_time, train_loss, valid_loss))

            if not debug_mode:
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

        np.save("./tmp/model/tr_losses_alpha{}_gamma{}.npy".format(alpha, gamma), train_losses)
        np.save("./tmp/model/va_losses_alpha{}_gamma{}.npy".format(alpha, gamma), valid_losses)

    else:
        print("Infer only mode -> skip training...")

    if not debug_mode:
        model.load_state_dict(torch.load(weights_path))

    logger = Logger()
    best_loss = 1e5

    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid)):
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

    logger.get_scores()
    if not infer_only:
        write_results(logger, args)

    show = 1
    if not infer_only:
        plt.figure(figsize=(14, 8))
        plt.plot(np.arange(num_epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(num_epochs), valid_losses, linestyle="--", label="valid")
        plt.legend()
        plt.savefig("./tmp/output/loss_batch{}_alpha{}_gamma{}.png".format(batch_size, alpha, gamma))

    plt.figure(figsize=(12, 10))
    plt.subplot(3,2,1)
    plt.imshow(best_original[0][show].cpu().numpy())

    if not infer_only:
        plt.subplot(3,2,2)
        plt.plot(np.arange(num_epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(num_epochs), valid_losses, linestyle="--", label="valid")
        plt.legend()
        
    plt.subplot(3,2,3)
    plt.imshow(valid_data.decode_segmt(torch.argmax(best_pred_segmt, dim=1)[show].cpu().numpy()))

    plt.subplot(3,2,4)
    plt.imshow(valid_data.decode_segmt(best_original[1][show].cpu().numpy()))

    plt.subplot(3,2,5)
    pred_clamped = torch.clamp(best_pred_depth, min=1e-9, max=1.)
    plt.imshow(pred_clamped[show].squeeze().cpu().numpy())

    plt.subplot(3,2,6)
    plt.imshow(best_original[2][show].squeeze().cpu().numpy())

    plt.tight_layout()
    ep_or_infer = "epoch{}-{}".format(save_at_epoch, num_epochs) if not infer_only else "infer-mode"
    plt.savefig("./tmp/output/xtask_" + ep_or_infer + "_batch{}_alpha{}_gamma{}.png".format(batch_size, alpha, gamma))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    img = ax1.imshow(np.abs((1 / best_pred_depth[show] - 1 / best_y_depth[show]).squeeze().cpu().numpy()))
    fig.colorbar(img, ax=ax1)
    plt.title("Absolute error of depth (non-inverted)")

    flat_pred = torch.flatten(best_pred_depth[show]).cpu().numpy()
    flat_targ = torch.flatten(best_y_depth[show]).cpu().numpy()
    sns.histplot(1 / flat_pred[flat_pred > 0], stat='density', color='blue', label='pred', ax=ax2)
    sns.histplot(1 / flat_targ[flat_targ > 0], stat='density', color='green', label='target', ax=ax2)
    plt.title("Density plot of depth (non-inverted")
    plt.legend()

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
    plt.savefig("./tmp/output/xtask_hist_batch{}_alpha{}_gamma{}.png".format(batch_size, alpha, gamma))
    
    # plt.show()