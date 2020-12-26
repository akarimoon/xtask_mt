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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataloader import NYUv2
from parser import nyu_xtask_parser
from module import Logger, XTaskLoss
from model.xtask_ts import XTaskTSNet
from utils import *

def compute_loss(batch_X, batch_y_segmt, batch_y_depth,
                 model, task_weights=None,
                 criterion=None, optimizer=None, 
                 is_train=True):

    if task_weights is None:
        task_weights = [1, 1]

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)

    output = model(batch_X)
    image_loss, label_loss = criterion(output, batch_y_segmt, batch_y_depth, task_weights=task_weights)

    if is_train:
        optimizer.zero_grad()
        image_loss.backward(retain_graph=True)
        label_loss.backward(retain_graph=True)
        optimizer.step()

    return (image_loss + label_loss).item()

if __name__ == '__main__':
    torch.manual_seed(0)
    opt = nyu_xtask_parser()
    opt.betas = (opt.b1, opt.b2)
    opt.num_classes = 13
    opt.gradnorm = False

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
    print("   using ResNet{}, optimizer: Adam (lr={}, beta={}), scheduler: StepLR({}, {})".format(
        opt.enc_layers, opt.lr, opt.betas, opt.scheduler_step_size, opt.scheduler_gamma))
    print("   loss function --- Lp_depth: {}, tsegmt: {}, tdepth: {}".format(
        opt.lp, opt.tseg_loss, opt.tdep_loss))
    print("   hyperparameters --- alpha: {}, gamma: {}, smoothing: {}".format(
        opt.alpha, opt.gamma, opt.label_smoothing))
    print("   batch size: {}, train for {} epochs".format(
        opt.batch_size, opt.epochs))
    print("   batch_norm={}, wider_ttnet={}".format(
        opt.batch_norm, opt.wider_ttnet))

    device_name = "cpu" if opt.cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=opt.enc_layers, 
                        out_features_segmt=opt.num_classes,
                        batch_norm=opt.batch_norm, wider_ttnet=opt.wider_ttnet,
                        use_pretrain=opt.use_pretrain
                        )
    model.to(device)
    parameters_to_train = [p for p in model.parameters()]
    print("Parameter Space: {}".format(count_parameters(model)))
    print("TransferNet type:")
    print("   {}".format(model.trans_name))
    
    print("Options:")
    task_weights = None
    if opt.uncertainty_weights:
        print("   use uncertainty weights")
        """
        Implementation of uncertainty weights (learnable weight parameters to balance losses of multiple tasks)
        See arxiv.org/abs/1705.07115
        """
        log_var_a = torch.zeros((1,), requires_grad=True, device=device_name)
        log_var_b = torch.zeros((1,), requires_grad=True, device=device_name)
        task_weights = [log_var_a, log_var_b]
        parameters_to_train += task_weights

    print("Loading dataset...")
    train_data = NYUv2(root_path=opt.input_path, split='train', transforms=True)
    valid_data = NYUv2(root_path=opt.input_path, split='val', transforms=True)

    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    valid = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    criterion = XTaskLoss(num_classes=opt.num_classes, 
                          alpha=opt.alpha, gamma=opt.gamma, label_smoothing=opt.label_smoothing,
                          image_loss_type=opt.lp, t_segmt_loss_type=opt.tseg_loss, t_depth_loss_type=opt.tdep_loss,
                          ignore_index=opt.ignore_index).to(device)
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
                batch_X, batch_y_segmt, batch_y_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth,
                                    model, task_weights=task_weights,
                                    criterion=criterion, optimizer=optimizer, is_train=True)
                train_loss += loss

            for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
                batch_X, batch_y_segmt, batch_y_depth = batch
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                    model, task_weights=task_weights,
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
                        (torch.exp(task_weights[1]) ** 0.5).item(), (torch.exp(task_weights[0]) ** 0.5).item()))

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

        np.save(os.path.join(results_dir, "model", "tr_losses.npy"), train_losses)
        np.save(os.path.join(results_dir, "model", "va_losses.npy"), valid_losses)

    else:
        save_at_epoch = 0
        print("Infer only mode -> skip training...")

    if not opt.debug:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    if opt.infer_only:
        try:
            train_losses = np.load(os.path.join(results_dir, "model", "tr_losses.npy"))
            valid_losses = np.load(os.path.join(results_dir, "model", "va_losses.npy"))
        except:
            train_losses = None
            valid_losses = None

    logger = Logger(num_classes=opt.num_classes, ignore_index=opt.ignore_index)
    best_score = 0
    best_set = {}

    model.eval()
    with torch.no_grad():
        # GPU warm-up
        if opt.infer_only:    
            dummy_input = torch.randn(1, 3, opt.height, opt.width, dtype=torch.float).to(device)
            for _ in range(100):
                _ = model(dummy_input)
        for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
            batch_X, batch_y_segmt, batch_y_depth = batch
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
            batch_y_depth = batch_y_depth.to(device, non_blocking=True)

            predicted = model(batch_X, infer_only=opt.time_inf)

            # image_loss, label_loss = criterion(predicted, batch_y_segmt, batch_y_depth)

            pred_segmt, pred_t_segmt, pred_depth, pred_t_depth = predicted

            preds = [pred_segmt, pred_depth]
            targets = [batch_y_segmt, batch_y_depth]
            logger.log(preds, targets)

            # use best results for final plot
            score = overall_score(preds, targets)
            if i == 0 or score > best_score:
                best_score = score
                best_set["score"] = score
                best_set["original"] = batch_X
                best_set["targ_segmt"] = batch_y_segmt
                best_set["targ_depth"] = batch_y_depth
                best_set["pred_segmt"] = pred_segmt
                best_set["pred_depth"] = pred_depth
                best_set["pred_tsegmt"] = pred_t_segmt
                best_set["pred_tdepth"] = pred_t_depth

    logger.get_scores()

    if not opt.infer_only:
        write_results(logger, opt, model, exp_num=exp_num)
        write_indv_results(opt, model, folder_path=results_dir)

    make_plots(opt, results_dir, best_set, save_at_epoch, valid_data, train_losses, valid_losses, is_nyu=True)