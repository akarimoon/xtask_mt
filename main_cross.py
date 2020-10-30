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
from parser import cityscapes_parser
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

if __name__=='__main__':
    torch.manual_seed(0)
    args = cityscapes_parser()

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
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_workers = args.workers
    weights_path = "./test.pth"

    infer_only = args.infer_only
    use_cpu = args.cpu
    debug_mode = args.debug

    device_name = "cpu" if use_cpu else "cuda"
    device = torch.device(device_name)
    print("device: {}".format(device))
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

    criterion = XTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.1)

    if not infer_only:
        print("=======================================")
        print("Start training...")
        best_valid_loss = 100.
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
                if valid_loss < best_valid_loss:
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
            if loss < best_loss:
                best_loss = loss
                best_original = original
                best_pred_segmt = pred_segmt
                best_pred_depth = pred_depth

    logger.get_scores()

    show = 1
    if not infer_only:
        plt.figure(figsize=(14, 8))
        plt.plot(np.arange(num_epochs), train_losses, linestyle="-", label="train")
        plt.plot(np.arange(num_epochs), valid_losses, linestyle="--", label="valid")
        plt.legend()
        plt.savefig("./tmp/output/loss_xtask.png")

    plt.figure(figsize=(12, 10))
    plt.subplot(3,2,1)
    plt.imshow(best_original[0][show].cpu().numpy())

    if not infer_only:
        plt.subplot(3,2,2)
        plt.figure(figsize=(14, 8))
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
    plt.savefig("./tmp/output/xtask_" + ep_or_infer + "_batch{}.png".format(batch_size))

    plt.figure(figsize=(8, 5))
    flat_pred = torch.flatten(best_pred_depth[show]).cpu().numpy()
    flat_targ = torch.flatten(best_original[2][show]).cpu().numpy()
    sns.histplot(1 / flat_pred[flat_pred > 0], stat='density', color='blue')
    sns.histplot(1 / flat_targ[flat_targ > 0], stat='density', color='green')
    plt.savefig("./tmp/output/xtask_hist.png")

    plt.show()