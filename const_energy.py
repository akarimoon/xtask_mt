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
from tqdm import tqdm

from dataloader import CityscapesDataset
from parser import const_energy_parser
from module import Logger, XTaskLoss
from model.xtask_ts import XTaskTSNet
from utils import *

DEPTH_CORRECTION = 2.1116e-09

def const_energy(diff_segmt, diff_depth):
    energy_segmt = torch.mean(torch.sum((diff_segmt - torch.mean(diff_segmt)) / torch.std(diff_segmt), dim=(1, 2, 3)))
    energy_depth = torch.mean(torch.sum((diff_depth - torch.mean(diff_depth)) / torch.std(diff_depth), dim=(1, 2, 3)))
    return 0.5 * energy_segmt + 0.5 * energy_depth

def compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                 batch_mask_segmt, batch_mask_depth, 
                 model, task_weights=None, use_xtc=False,
                 criterion=None, optimizer=None, 
                 is_train=True):

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)
    batch_mask_segmt = batch_mask_segmt.to(device, non_blocking=True)
    batch_mask_depth = batch_mask_depth.to(device, non_blocking=True)

    output = model(batch_X)
    image_loss, label_loss = criterion(output, batch_y_segmt, batch_y_depth,
                                       batch_mask_segmt, batch_mask_depth, task_weights=task_weights, use_xtc=use_xtc)

    if is_train:
        optimizer.zero_grad()
        image_loss.backward(retain_graph=True)
        label_loss.backward(retain_graph=True)
        optimizer.step()

    return (image_loss + label_loss).item(), output

if __name__=='__main__':
    torch.manual_seed(0)
    opt = const_energy_parser()
    opt.use_xtc = True

    print("Initializing...")
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not opt.use_xtc:
        weights_path = os.path.join(opt.save_path, "energy_xtsc.pth")
    else:
        weights_path = os.path.join(opt.save_path, "energy_xtc.pth")

    parameters_to_train = []

    device_name = "cpu" if opt.cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=34, out_features_segmt=7)
    model.to(device)

    parameters_to_train = [p for p in model.parameters()]
    
    print("Options:")
    task_weights = None
    opt.balance_method = None
    if opt.uncertainty_weights:
        print("   use uncertainty weights")
        opt.balance_method = "uncert"
        """
        Implementation of uncertainty weights (learnable weight parameters to balance losses of multiple tasks)
        See arxiv.org/abs/1705.07115
        """
        log_var_a = torch.zeros((1,), requires_grad=True, device=device_name)
        log_var_b = torch.zeros((1,), requires_grad=True, device=device_name)
        task_weights = [log_var_a, log_var_b]
        parameters_to_train += task_weights

    print("Loading dataset...")
    train_data = CityscapesDataset(root_path=opt.input_path, height=128, width=256, num_classes=7,
                                   split='train', transform=["random_flip", "random_crop"])
    valid_data = CityscapesDataset(root_path=opt.input_path, height=128, width=256, num_classes=7,
                                   split='val', transform=None)
    # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    train = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=opt.workers)
    valid = DataLoader(valid_data, batch_size=8, shuffle=True, num_workers=opt.workers)

    criterion = XTaskLoss(num_classes=7, balance_method=opt.balance_method).to(device)

    optimizer = optim.Adam(parameters_to_train, lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    if not opt.infer_only:
    print("=======================================")
    print("Start training...")
    best_valid_loss = 1e5
    valid_losses = []
    energy_hist = []
    save_at_epoch = 0

        for epoch in range(1, opt.epochs + 1):

            start = time.time()
            valid_loss = 0.

            for i, batch in enumerate(tqdm(train, disable=opt.notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                _, output = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                    batch_mask_segmt, batch_mask_depth, 
                                    model, task_weights=task_weights,
                                    criterion=criterion, optimizer=optimizer, use_xtc=opt.use_xtc,
                                    is_train=True)

            for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                loss, output = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                        batch_mask_segmt, batch_mask_depth, 
                                        model, task_weights=task_weights,
                                        criterion=criterion, optimizer=optimizer, use_xtc=opt.use_xtc,
                                        is_train=False)
                valid_loss += loss

                if i == 0:
                    diff_segmt = torch.abs(F.log_softmax(output[0].cpu().detach(), dim=1) -\
                                        F.log_softmax(output[1].cpu().detach(), dim=1))
                    diff_depth = torch.abs(output[2].cpu().detach() - output[3].cpu().detach())

                else:
                    d_seg = torch.abs(F.log_softmax(output[0].cpu().detach(), dim=1) -\
                                    F.log_softmax(output[1].cpu().detach(), dim=1))
                    d_dep = torch.abs(output[2].cpu().detach() - output[3].cpu().detach())
                    diff_segmt = torch.cat([diff_segmt, d_seg])
                    diff_depth = torch.cat([diff_depth, d_dep])

            valid_loss /= len(valid.dataset)
            valid_losses.append(valid_loss)

            energy = const_energy(diff_segmt, diff_depth)
            energy_hist.append(energy.numpy())

            elapsed_time = (time.time() - start) / 60
            print("Epoch {}/{} [{:.1f}min] --- valid loss: {:.5f} --- const energy: {:.5f}".format(
                        epoch, opt.epochs, elapsed_time, valid_loss, energy))

            print("Saving weights...")
            weights = model.state_dict()
            torch.save(weights, weights_path)

            scheduler.step()

        print("Training done")
        print("=======================================")

        valid_losses = np.array(valid_losses)
        energy_hist = np.array(energy_hist)

        if not opt.use_xtc:
            np.save(os.path.join(opt.save_path, "xtsc_va_losses.npy"), valid_losses)
            np.save(os.path.join(opt.save_path, "xtsc_energy.npy"), energy_hist)
        else:
            np.save(os.path.join(opt.save_path, "xtc_va_losses.npy"), valid_losses)
            np.save(os.path.join(opt.save_path, "xtc_energy.npy"), energy_hist)

    else:
        xtsc_loss = np.load(os.path.join(opt.save_path, "xtsc_va_losses.npy"))
        xtc_loss = np.load(os.path.join(opt.save_path, "xtc_va_losses.npy"))
        xtsc_energy = np.load(os.path.join(opt.save_path, "xtsc_energy.npy"))
        xtc_energy = np.load(os.path.join(opt.save_path, "xtc_energy.npy"))

        eps = len(xtsc_loss)
        
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(np.arange(eps), xtsc_loss, color='blue', label='xtsc')
        plt.plot(np.arange(eps), xtc_loss, color='green', label='xtc')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')

        plt.subplot(1,2,2)
        plt.plot(np.arange(eps), xtsc_energy, color='blue', label='xtsc')
        plt.plot(np.arange(eps), xtc_energy, color='green', label='xtc')
        plt.xlabel('Epoch')
        plt.ylabel('Energy')
        plt.title('Energy History')

        plt.legend()
        plt.tight_layout()
        plt.show()