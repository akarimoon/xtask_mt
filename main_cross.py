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
            print("Epoch {}/{} [{:.1f}min] ({}) --- train loss: {:.5f} --- valid loss: {:.5f}".format(
                        epoch, num_epochs, elapsed_time, mode, train_loss, valid_loss))

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