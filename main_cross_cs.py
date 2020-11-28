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
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.manual_seed(0)
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
    print("   loss function --- Lp_depth: {}, tsegmt: {}, tdepth: {}".format(
        opt.lp, opt.tseg_loss, opt.tdep_loss))
    print("   hyperparameters --- alpha: {}, gamma: {}, temp:{}, smoothing: {}".format(
        opt.alpha, opt.gamma, opt.temp, opt.label_smoothing))
    print("   batch size: {}, train for {} epochs".format(
        opt.batch_size, opt.epochs))

    device_name = "cpu" if opt.cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=opt.enc_layers, 
                       out_features_segmt=opt.num_classes, 
                       is_shallow=opt.is_shallow, 
                       batch_norm=opt.batch_norm, 
                       use_pretrain=opt.use_pretrain
                    )
    model.to(device)

    model = torch.nn.DataParallel(model) # make parallel
    # cudnn.benchmark = True

    parameters_to_train = [p for p in model.parameters()]
    print("TransferNet type:")
    print("   {}".format(model.module.trans_name))
    
    print("Options:")
    log_vars = None
    if opt.is_shallow:
        print("   shallow decoder")
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
                                   split='train', transform=["random_flip", "random_crop"])
    valid_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                   split='val', transform=None)
    # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    valid = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    criterion = XTaskLoss(num_classes=opt.num_classes, 
                          alpha=opt.alpha, gamma=opt.gamma, temp=opt.temp, label_smoothing=opt.label_smoothing,
                          image_loss_type=opt.lp, t_segmt_loss_type=opt.tseg_loss, t_depth_loss_type=opt.tdep_loss,
                          grad_loss=opt.grad_loss).to(device)
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
                    weights = model.module.state_dict()
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
        train_losses = None
        valid_losses = None
        save_at_epoch = 0
        print("Infer only mode -> skip training...")

    if not opt.debug:
        model.module.load_state_dict(torch.load(weights_path, map_location=device))

    logger = Logger(num_classes=opt.num_classes)
    best_loss = 1e5
    best_set = {}

    model.eval()
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
            loss = overall_score(preds, targets)
            if i == 0 or loss < best_loss:
                best_set["loss"] = loss
                best_set["original"] = original
                best_set["targ_segmt"] = batch_y_segmt
                best_set["targ_depth"] = batch_y_depth
                best_set["pred_segmt"] = pred_segmt
                best_set["pred_depth"] = pred_depth
                best_set["pred_tsegmt"] = pred_t_segmt
                best_set["pred_tdepth"] = pred_t_depth

    logger.get_scores()

    if not opt.infer_only:
        write_results(logger, opt, model.module, exp_num=exp_num)
        write_indv_results(opt, model.module, folder_path=results_dir)

    make_plots(opt, results_dir, best_set, save_at_epoch, valid_data, train_losses, valid_losses)