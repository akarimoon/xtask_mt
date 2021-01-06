import argparse, sys, os, time
import cv2
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import CityscapesDataset, NYUv2
from parser import stl_parser
from module import STLLogger
from model.xtask_ts import BaseSTLNet
from utils import *

DEPTH_CORRECTION = 2.1116e-09

def compute_loss(batch_X, batch_y, batch_mask, 
                 model, data, task,
                 criterion=None, optimizer=None, 
                 is_train=True):

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    if batch_mask is not None:
        batch_mask = batch_mask.to(device, non_blocking=True)

    output = model(batch_X)
    if data == 'cs':
        if task == 'segmt':
            loss = criterion(output, batch_y)
        elif task == 'depth':
            loss = criterion(output, batch_y, batch_mask)
    elif data == 'nyu':
        loss = criterion(output, batch_y)

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def masked_L1_loss(predicted, target, mask):
    diff = torch.abs(predicted - target) * mask
    loss = torch.sum(diff, dim=(2,3)) / torch.sum(mask, dim=(2,3))
    return torch.mean(loss)

if __name__=='__main__':
    torch.manual_seed(0)
    opt = stl_parser()
    if opt.data == 'cs':
        opt.input_path = './data/cityscapes'
        opt.num_classes = 7
        opt.out_features = opt.num_classes if opt.task == 'segmt' else 1
        opt.height = 128
        opt.width = 256
        opt.ignore_index = 250
        opt.batch_size = 8
        opt.epochs = 250
        opt.scheduler_step_size = 80
    elif opt.data == 'nyu':
        opt.input_path = './data/nyu'
        opt.num_classes = 13
        opt.out_features = opt.num_classes if opt.task == 'segmt' else 1
        opt.height = 288
        opt.width = 384
        opt.ignore_index = -1
        opt.batch_size = 6
        opt.epochs = 100
        opt.scheduler_step_size = 60

    print("STL for {} (task: {})".format(opt.data, opt.task))

    print("Initializing...")
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    weights_path = os.path.join(opt.save_path, "model", "stl_{}_{}.pth".format(opt.data, opt.task))

    parameters_to_train = []

    device_name = "cpu" if opt.cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = BaseSTLNet(enc_layers=34, out_features=opt.out_features)
    model.to(device)
    parameters_to_train = [p for p in model.parameters()]
    print("Parameter Space: {}".format(count_parameters(model)))
    
    print("Loading dataset...")

    if opt.data == 'cs':
        train_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                    split='train', transform=["random_flip", "random_crop"], ignore_index=opt.ignore_index)
        valid_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                    split='val', transform=None, ignore_index=opt.ignore_index)
        # test_data = CityscapesDataset('./data/cityscapes', split='train', transform=transform)
    elif opt.data == 'nyu':
        train_data = NYUv2(root_path=opt.input_path, split='train', transforms=True)
        valid_data = NYUv2(root_path=opt.input_path, split='val', transforms=None)

    train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    if opt.data == 'cs':
        criterion = nn.CrossEntropyLoss(ignore_index=opt.ignore_index) if opt.task == 'segmt' else masked_L1_loss
    elif opt.data == 'nyu':
        criterion = nn.CrossEntropyLoss(ignore_index=opt.ignore_index) if opt.task == 'segmt' else nn.L1Loss()

    optimizer = optim.Adam(parameters_to_train, lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.scheduler_step_size, gamma=0.5)

    print("=======================================")
    print("Start training...")
    best_valid_loss = 1e5

    for epoch in range(1, opt.epochs + 1):

        start = time.time()
        train_loss = 0.
        valid_loss = 0.

        for i, batch in enumerate(tqdm(train, disable=opt.notqdm)):
            if opt.data == 'cs':
                if opt.task == 'segmt':
                    _, batch_X, batch_y, _, batch_mask, _ = batch
                elif opt.task == 'depth':
                    _, batch_X, _, batch_y, _, batch_mask = batch
            elif opt.data == 'nyu':
                if opt.task == 'segmt':
                    batch_X, batch_y, _ = batch
                elif opt.task == 'depth':
                    batch_X, _, batch_y = batch
                batch_mask = None
            loss = compute_loss(batch_X, batch_y, batch_mask, 
                                model, data=opt.data, task=opt.task,
                                criterion=criterion, optimizer=optimizer,
                                is_train=True)
            train_loss += loss

        for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
            if opt.data == 'cs':
                if opt.task == 'segmt':
                    _, batch_X, batch_y, _, batch_mask, _ = batch
                elif opt.task == 'depth':
                    _, batch_X, _, batch_y, _, batch_mask = batch
            elif opt.data == 'nyu':
                if opt.task == 'segmt':
                    batch_X, batch_y, _ = batch
                elif opt.task == 'depth':
                    batch_X, _, batch_y = batch
                batch_mask = None
            loss = compute_loss(batch_X, batch_y, batch_mask, 
                                model, data=opt.data, task=opt.task,
                                criterion=criterion, optimizer=optimizer,
                                is_train=False)
            valid_loss += loss

        train_loss /= len(train.dataset)
        valid_loss /= len(valid.dataset)

        elapsed_time = (time.time() - start) / 60
        print("Epoch {}/{} [{:.1f}min] --- train loss: {:.5f} --- valid loss: {:.5f}".format(
                    epoch, opt.epochs, elapsed_time, train_loss, valid_loss))

        if not opt.debug:
            if epoch == 0 or valid_loss < best_valid_loss:
                print("Saving weights...")
                weights = model.state_dict()
                torch.save(weights, weights_path)
                best_valid_loss = valid_loss

        scheduler.step()

    print("Training done")
    print("=======================================")

    model.load_state_dict(torch.load(weights_path, map_location=device))

    logger = STLLogger(task=opt.task, num_classes=opt.num_classes, ignore_index=opt.ignore_index)
    best_score = 0
    best_set = {}

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
            if opt.data == 'cs':
                if opt.task == 'segmt':
                    _, batch_X, batch_y, _, batch_mask, _ = batch
                elif opt.task == 'depth':
                    _, batch_X, _, batch_y, _, batch_mask = batch
            elif opt.data == 'nyu':
                if opt.task == 'segmt':
                    batch_X, batch_y, _ = batch
                elif opt.task == 'depth':
                    batch_X, _, batch_y = batch

            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y_segmt.to(device, non_blocking=True)

            pred = model(batch_X)

            logger.log(pred, batch_y, batch_mask)

    logger.get_scores()