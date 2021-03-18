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
from parser import cityscapes_xtask_parser
from module import Logger, XTaskLoss
from model.xtask_ts import XTaskTSNet
from utils import *

DEPTH_CORRECTION = 2.1116e-09

def compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                 batch_mask_segmt, batch_mask_depth, 
                 model, task_weights=None,
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
                                       batch_mask_segmt, batch_mask_depth, task_weights=task_weights)

    if is_train:
        optimizer.zero_grad()
        image_loss.backward(retain_graph=True)
        label_loss.backward(retain_graph=True)
        optimizer.step()

    return (image_loss + label_loss).item()

def compute_loss_with_gradnorm(batch_X, batch_y_segmt, batch_y_depth, 
                               batch_mask_segmt, batch_mask_depth, 
                               model, task_weights=None, l01=None, l02=None,
                               criterion=None, criterion2=None, optimizer=None, optimizer2=None, 
                               is_train=True, epoch=1):

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)
    batch_mask_segmt = batch_mask_segmt.to(device, non_blocking=True)
    batch_mask_depth = batch_mask_depth.to(device, non_blocking=True)

    output = model(batch_X)
    image_loss, label_loss = criterion(output, batch_y_segmt, batch_y_depth,
                                       batch_mask_segmt, batch_mask_depth, task_weights=task_weights)

    if is_train:

        alpha = 0.16

        l1 = task_weights[0] * image_loss * 0.5
        l2 = task_weights[1] * label_loss * 0.5

        if epoch == 1:
            l01 = l1.data
            l02 = l2.data

        optimizer.zero_grad()
        l1.backward(retain_graph=True)
        l2.backward(retain_graph=True)
            
        param = list(model.pretrained_encoder.layer4[-1].conv2.parameters())
        G1R = torch.autograd.grad(l1, param[0], retain_graph=True, create_graph=True)
        G1 = torch.norm(G1R[0], 2)
        G2R = torch.autograd.grad(l2, param[0], retain_graph=True, create_graph=True)
        G2 = torch.norm(G2R[0], 2)
        G_avg = (G1 + G2) / 2
        
        # Calculating relative losses 
        lhat1 = torch.div(l1, l01)
        lhat2 = torch.div(l2, l02)
        lhat_avg = (lhat1 + lhat2) / 2
        
        # Calculating relative inverse training rates for tasks 
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)
        
        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = G_avg * inv_rate1 ** alpha
        C2 = G_avg * inv_rate2 ** alpha
        C1 = C1.detach()
        C2 = C2.detach()
        
        optimizer2.zero_grad()
        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
        Lgrad = torch.add(criterion2(G1, C1), criterion2(G2, C2))
        Lgrad.backward()
        
        # Updating loss weights 
        optimizer2.step()

        optimizer.step()

    return (task_weights[0] * image_loss).item() + (task_weights[1] * label_loss).item(), l01, l02

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
    if opt.optim == 'adam':
        print("   using ResNet{}, optimizer: Adam (lr={}, beta={}), scheduler: StepLR({}, {})".format(
            opt.enc_layers, opt.lr, opt.betas, opt.scheduler_step_size, opt.scheduler_gamma))
    elif opt.optim == 'sgd':
        print("   using ResNet{}, optimizer: SGD (lr={}), scheduler: StepLR({}, {})".format(
            opt.enc_layers, opt.lr, opt.scheduler_step_size, opt.scheduler_gamma))
    print("   loss function --- Lp_depth: {}, tsegmt: {}, tdepth: {}".format(
        opt.lp, opt.tseg_loss, opt.tdep_loss))
    print("   hyperparameters --- lambda_2: {}, lambda_1: {}, smoothing: {}".format(
        opt.lambda_2, opt.lambda_1, opt.label_smoothing))
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
    opt.balance_method = None
    if opt.multiple_gpu:
        print("   multiple GPUs")
        model = torch.nn.DataParallel(model)
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
    if opt.gradnorm:
        print("   use gradnorm")
        opt.balance_method = "gradnorm"
        w_loss_1 = torch.tensor(1, requires_grad=True, dtype=torch.float32, device=device_name)
        w_loss_2 = torch.tensor(1, requires_grad=True, dtype=torch.float32, device=device_name)
        task_weights = [w_loss_1, w_loss_2]
        optimizer2 = optim.Adam(task_weights, lr=opt.lr)
        criterion2 = nn.L1Loss()
        l01, l02 = None, None

    print("Loading dataset...")
    train_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                   split='train', transform=["random_flip", "random_crop"], ignore_index=opt.ignore_index)
    valid_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                   split='val', transform=None, ignore_index=opt.ignore_index)
    # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    valid = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    criterion = XTaskLoss(num_classes=opt.num_classes, 
                          lambda_2=opt.lambda_2, lambda_1=opt.lambda_1, label_smoothing=opt.label_smoothing,
                          image_loss_type=opt.lp, t_segmt_loss_type=opt.tseg_loss, t_depth_loss_type=opt.tdep_loss,
                          balance_method=opt.balance_method,
                          ignore_index=opt.ignore_index).to(device)
    if opt.optim == 'adam':
        optimizer = optim.Adam(parameters_to_train, lr=opt.lr, betas=opt.betas)
    elif opt.optim == 'sgd':
        optimizer = optim.SGD(parameters_to_train, lr=opt.lr)
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
                if not opt.gradnorm:
                    loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                        batch_mask_segmt, batch_mask_depth, 
                                        model, task_weights=task_weights,
                                        criterion=criterion, optimizer=optimizer,
                                        is_train=True)
                else:
                    loss, l01, l02 = compute_loss_with_gradnorm(batch_X, batch_y_segmt, batch_y_depth, 
                                        batch_mask_segmt, batch_mask_depth, 
                                        model, task_weights=task_weights, l01=l01, l02=l02,
                                        criterion=criterion, criterion2=criterion2, optimizer=optimizer, optimizer2=optimizer2,
                                        is_train=True, epoch=epoch)
                    coef = 2 / (w_loss_1 + w_loss_2)
                    task_weights = [coef * w_loss_1, coef * w_loss_2]

                train_loss += loss

            for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
                _, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
                if not opt.gradnorm:
                    loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, 
                                        batch_mask_segmt, batch_mask_depth, 
                                        model, task_weights=task_weights,
                                        criterion=criterion, optimizer=optimizer,
                                        is_train=False)
                else:
                    loss, l01, l02 = compute_loss_with_gradnorm(batch_X, batch_y_segmt, batch_y_depth, 
                                        batch_mask_segmt, batch_mask_depth, 
                                        model, task_weights=task_weights, l01=l01, l02=l02,
                                        criterion=criterion, criterion2=criterion2, optimizer=optimizer, optimizer2=optimizer2,
                                        is_train=False, epoch=epoch)
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
            if opt.gradnorm:
                print("GradNorm task weights: segmt={:.5f}, depth={:.5f}".format(
                        task_weights[1].item(), task_weights[0].item()))

            if not opt.debug:
                if epoch == 0 or valid_loss < best_valid_loss:
                    print("Saving weights...")
                    if not opt.multiple_gpu:
                        weights = model.state_dict()
                    else:
                        weights = model.module.state_dict()
                    torch.save(weights, weights_path)
                    best_valid_loss = valid_loss
                    save_at_epoch = epoch

            scheduler.step()

        print("Training done")
        print("=======================================")

        train_losses = np.array(train_losses)
        valid_losses = np.array(valid_losses)

        np.save(os.path.join(results_dir, "model", "tr_losses.npy".format(opt.lambda_2, opt.lambda_1)), train_losses)
        np.save(os.path.join(results_dir, "model", "va_losses.npy".format(opt.lambda_2, opt.lambda_1)), valid_losses)

    else:
        save_at_epoch = 0
        print("Infer only mode -> skip training...")

    if not opt.debug:
        if not opt.multiple_gpu:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            model.module.load_state_dict(torch.load(weights_path, map_location=device))
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
            original, batch_X, batch_y_segmt, batch_y_depth, batch_mask_segmt, batch_mask_depth = batch
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
            batch_y_depth = batch_y_depth.to(device, non_blocking=True)
            batch_mask_segmt = batch_mask_segmt.to(device, non_blocking=True)
            batch_mask_depth = batch_mask_depth.to(device, non_blocking=True)

            predicted = model(batch_X, direct_only=opt.direct_only)
            # _, _ = criterion(predicted, batch_y_segmt, batch_y_depth,
            #                                    batch_mask_segmt, batch_mask_depth)

            pred_segmt, pred_t_segmt, pred_depth, pred_t_depth = predicted

            preds = [pred_segmt, pred_depth]
            targets = [batch_y_segmt, batch_y_depth]
            masks = [batch_mask_segmt, batch_mask_depth]
            logger.log(preds, targets, masks)

            # use best results for final plot
            score = overall_score(preds, targets)
            if i == 0 or score > best_score:
                best_score = score
                best_set["score"] = score
                best_set["original"] = original
                best_set["targ_segmt"] = batch_y_segmt
                best_set["targ_depth"] = batch_y_depth
                best_set["pred_segmt"] = pred_segmt
                best_set["pred_depth"] = pred_depth
                best_set["pred_tsegmt"] = pred_t_segmt
                best_set["pred_tdepth"] = pred_t_depth

    logger.get_scores()

    if not opt.infer_only:
        if not opt.multiple_gpu:
            write_results(logger, opt, model, exp_num=exp_num)
            write_indv_results(opt, model, folder_path=results_dir)
        else:
            write_results(logger, opt, model.module, exp_num=exp_num)
            write_indv_results(opt, model.module, folder_path=results_dir)

    make_plots(opt, results_dir, best_set, save_at_epoch, valid_data, train_losses, valid_losses)