import os
import torch
import fnmatch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
import shutil

from dataset.nyuv2 import *
from dataset.cityscapes import *
from torch.autograd import Variable
from model.mtan import SegNet2tasks as SegNet
from model.mtan_single import SegNet as SegNet_STAN

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from progress.bar import Bar as Bar
import pdb

parser = argparse.ArgumentParser(description='Knowledge Distillation for Multi-task Learning (MTAN) for 2 tasks')
parser.add_argument('--dataset', default='nyu', type=str, help='choose dataset: nyu, cs')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--alr', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation')
parser.add_argument('--single-dir', default='/home/s1798461/Code/mtan-results/im2im_pred/', help='Directory to output the result')
opt = parser.parse_args()

def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, '{}_mtan_kdmtl_'.format(opt.dataset) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, '{}_mtan_kdmtl_'.format(opt.dataset) + 'model_best.pth.tar'))

class transformer(torch.nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 512, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, bias=False)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs[0]))
        results.append(self.conv2(inputs[1]))
        return results

# define model, optimiser and scheduler
tasks = ['semantic', 'depth']

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
title = 'NYUv2' if opt.dataset == 'nyu' else 'CS'
logger = Logger(os.path.join(opt.out, 'mtan_kdmtl_' + 'log.txt'), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'ds', 'dd', 'dn'])


# define model, optimiser and scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
use_cuda = torch.cuda.is_available()

class_nb = 13 if opt.dataset == 'nyu' else 7
ignore_index = -1 if opt.dataset == 'nyu' else 250

model = SegNet(ignore_index=ignore_index).cuda()
single_model = {}
transformers = {}
for i, t in enumerate(tasks):
    single_model[i] = SegNet_STAN(class_nb=class_nb, task=tasks[i], ignore_index=ignore_index).cuda()
    checkpoint = torch.load(os.path.join(opt.out, '{}_mtan_single_model_task_{}_model_best.pth.tar'.format(opt.dataset, tasks[i])))
    single_model[i].load_state_dict(checkpoint['state_dict'])
    transformers[i] = transformer().cuda()


optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
params = []
for i in range(len(tasks)):
    params += transformers[i].parameters()
transformer_optimizer = optim.Adam(params, lr=opt.alr, weight_decay=5e-4)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR\n')

# define dataset path
dataset_path = opt.dataroot
if opt.dataset == 'nyu':
    train_set = NYUv2(root=dataset_path, train=True, augmentation=opt.apply_augmentation)
    test_set = NYUv2(root=dataset_path, train=False, augmentation=opt.apply_augmentation)
elif opt.dataset == 'cs':
    if opt.apply_augmentation:
        train_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='train',
                                        transform=['random_flip', 'random_crop'], ignore_index=ignore_index)
    else:
        train_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='train',
                                        transform=None, ignore_index=ignore_index)
    test_set = MyCityscapesDataset(height=128, width=256, root_path=dataset_path, num_classes=7, split='val',
                                    transform=None, ignore_index=ignore_index)

batch_size = 2
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=True)


# define parameters
total_epoch = 200
train_batch = len(train_loader)
test_batch = len(test_loader)
avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
best_loss = 100
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(12, dtype=np.float32)
    scheduler.step()
    dist_loss_save = {}
    for i, t in enumerate(tasks):
        dist_loss_save[i] = AverageMeter()

    # iteration for all batches
    model.train()
    train_dataset = iter(train_loader)
    bar = Bar('Training', max=train_batch)
    for k in range(train_batch):
        if opt.dataset == 'nyu':
            train_data, train_label, train_depth, _ = train_dataset.next()
        else:
            train_data, train_label, train_depth = train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth = train_depth.cuda()

        train_pred, logsigma, feat_s = model(train_data)

        
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth)

        w = torch.ones(len(tasks)).cuda()
        loss = torch.mean(sum(w[i] * train_loss[i] for i in range(3)))


        dist_loss = []
        # pdb.set_trace()
        for i, t in enumerate(tasks):
            with torch.no_grad():
                _, feat_ti = single_model[i](train_data)
            feat_ti0 = feat_ti[0].detach()
            feat_ti0 = feat_ti0 / (feat_ti0.pow(2).sum(1) + 1e-6).sqrt().view(feat_ti0.size(0), 1, feat_ti0.size(2), feat_ti0.size(3))
            feat_ti1 = feat_ti[1].detach()
            feat_ti1 = feat_ti1 / (feat_ti1.pow(2).sum(1) + 1e-6).sqrt().view(feat_ti1.size(0), 1, feat_ti1.size(2), feat_ti1.size(3))
            feat_si = transformers[i](feat_s[i])
            # pdb.set_trace()
            feat_si0 = feat_si[0]
            feat_si0 = feat_si0 / (feat_si0.pow(2).sum(1) + 1e-6).sqrt().view(feat_si0.size(0), 1, feat_si0.size(2), feat_si0.size(3))
            feat_si1 = feat_si[1]
            feat_si1 = feat_si1 / (feat_si1.pow(2).sum(1) + 1e-6).sqrt().view(feat_si1.size(0), 1, feat_si1.size(2), feat_si1.size(3))
            dist_0 = (feat_si0 - feat_ti0).pow(2).sum(1).mean()
            dist_1 = (feat_si1 - feat_ti1).pow(2).sum(1).mean()
            dist_loss.append(dist_0 + dist_1)
            dist_loss_save[i].update(dist_loss[i].data.item(), feat_si[0].size(0))


        lambda_ = [1, 1]
        dist_loss = sum(dist_loss[i] * lambda_[i] for i in range(len(tasks)))
        loss = loss + dist_loss

        optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        transformer_optimizer.step()

        cost[0] = train_loss[0].item()
        cost[1] = model.compute_miou(train_pred[0], train_label).item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        avg_cost[index, :6] += cost[:6] / train_batch
        bar.suffix  = '({batch}/{size}) | LossS: {loss_s:.4f} | LossD: {loss_d:.4f} | ds: {ds:.4f} | dd: {dd:.4f}|'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_s=cost[1],
                    loss_d=cost[3],
                    ds=dist_loss_save[0].val,
                    dd=dist_loss_save[1].val
                    )
        bar.next()
    bar.finish()

    loss_index = (avg_cost[index, 0] + avg_cost[index, 3]) / 2.0
    isbest = loss_index < best_loss

    # evaluating test data
    model.eval()
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            if opt.dataset == 'nyu':
                test_data, test_label, test_depth, _ = test_dataset.next()
            else:
                test_data, test_label, test_depth = test_dataset.next()
            test_data, test_label = test_data.cuda(),  test_label.type(torch.LongTensor).cuda()
            test_depth = test_depth.cuda()

            test_pred, _, _ = model(test_data)
            test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth)

            cost[6] = test_loss[0].item()
            cost[7] = model.compute_miou(test_pred[0], test_label).item()
            cost[8] = model.compute_iou(test_pred[0], test_label).item()
            cost[9] = test_loss[1].item()
            cost[10], cost[11] = model.depth_error(test_pred[1], test_depth)

            avg_cost[index, 6:] += cost[6:] / test_batch


    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} |'
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11]))
    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11],
                dist_loss_save[0].avg, dist_loss_save[1].avg)
    if isbest:
        best_loss = loss_index
        print_index = index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 
print('The best results is:')
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} |'
              .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11]))