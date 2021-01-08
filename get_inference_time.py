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
from parser import time_inference_parser
from model.xtask_ts import XTaskTSNet
from utils import *

if __name__=='__main__':
    torch.manual_seed(0)
    opt = time_inference_parser()
    if opt.data == 'cs':
        opt.input_path = './data/cityscapes'
        opt.num_classes = 7
        opt.height = 128
        opt.width = 256
        opt.ignore_index = 250
    elif opt.data == 'nyu':
        opt.input_path = './data/nyu'
        opt.num_classes = 13
        opt.height = 288
        opt.width = 384
        opt.ignore_index = -1

    print("Initializing for {} dataset...".format(opt.data))
    exp_num = str(opt.exp_num).zfill(3)
    results_dir = os.path.join(opt.save_path, exp_num)
    weights_path = os.path.join(results_dir, "model", "model.pth")

    parameters_to_train = []

    print("Parameters:")
    print("   predicting at size [{}*{}]".format(opt.height, opt.width))
    print("   encoder: ResNet{}, batch_norm={}, wider_ttnet={}".format(
        opt.enc_layers, opt.batch_norm, opt.wider_ttnet))

    device_name = "cpu" if opt.cpu else "cuda"
    device = torch.device(device_name)
    print("   device: {}".format(device))
    model = XTaskTSNet(enc_layers=opt.enc_layers, 
                       out_features_segmt=opt.num_classes,
                       batch_norm=opt.batch_norm, wider_ttnet=opt.wider_ttnet,
                       use_pretrain=opt.use_pretrain
                    )
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    parameters_to_train = [p for p in model.parameters()]
    print("Parameter Space: {}".format(count_parameters(model)))
    print("TransferNet type:")
    print("   {}".format(model.trans_name))

    print("Loading dataset...")
    if opt.data == 'cs':
        valid_data = CityscapesDataset(root_path=opt.input_path, height=opt.height, width=opt.width, num_classes=opt.num_classes,
                                    split='val', transform=None, ignore_index=opt.ignore_index)
    elif opt.data == 'nyu':
        valid_data = NYUv2(root_path=opt.input_path, split='val', transforms=True)
    valid = DataLoader(valid_data, batch_size=1, shuffle=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed_times = []
    records = []

    # GPU warm-up
    dummy_input = torch.randn(1, 3, opt.height, opt.width, dtype=torch.float).to(device)
    for _ in range(100):
        _ = model(dummy_input)

    model.eval()
    with torch.no_grad():
        for j in range(opt.num):
            for i, batch in enumerate(tqdm(valid, disable=opt.notqdm)):
                if opt.data == 'cs':
                    _, batch_X, _, _, _, _ = batch
                elif opt.data == 'nyu':
                    batch_X, _, _ = batch
                batch_X = batch_X.to(device, non_blocking=True)

                start.record()
                predicted = model(batch_X, direct_only=opt.direct_only)
                end.record()

                torch.cuda.synchronize()
                batch_time = start.elapsed_time(end)
                elapsed_times.append(batch_time)
                
            print("Avg inference time: {:.3f}[ms]".format(np.mean(elapsed_times)))
            records.append(np.mean(elapsed_times))

            del batch_X
            del predicted

    print("Avg of {} runs: {:.3f}[ms]".format(opt.num, np.mean(records)))
