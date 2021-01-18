import argparse, time
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import AdaMTNet3tasks
from create_dataset import NYUv2
from logger import Logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_loss(batch_X, batch_y_segmt, batch_y_depth, batch_y_normal,
                 model,
                 criteria=None, optimizer=None, 
                 is_train=True):

    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)
    batch_y_normal = batch_y_normal.to(device, non_blocking=True)

    output = model(batch_X)
    segmt_loss = criteria[0](output[0], batch_y_segmt)
    depth_loss = criteria[1](output[1], batch_y_depth)
    normal_loss = criteria[1](output[2], batch_y_normal)

    if is_train:
        optimizer.zero_grad()
        segmt_loss.backward(retain_graph=True)
        depth_loss.backward(retain_graph=True)
        normal_loss.backward(retain_graph=True)
        optimizer.step()

    return (segmt_loss + depth_loss + normal_loss).item()

def compute_loss_with_adalsp(batch_X, batch_y_segmt, batch_y_depth, batch_y_normal,
                             model,
                             criteria=None, optimizer=None, 
                             is_train=True):
    
    model.train(is_train)

    batch_X = batch_X.to(device, non_blocking=True)
    batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
    batch_y_depth = batch_y_depth.to(device, non_blocking=True)
    batch_y_normal = batch_y_normal.to(device, non_blocking=True)

    output = model(batch_X)
    segmt_loss = criteria[0](output[0], batch_y_segmt)
    depth_loss = criteria[1](output[1], batch_y_depth)
    normal_loss = criteria[1](output[2], batch_y_normal)

    if is_train:
        enc_g1 = []
        enc_g2 = []
        enc_g3 = []

        optimizer.zero_grad()
        segmt_loss.backward(retain_graph=True)
        for p in model.encoder.parameters():
            enc_g1.append(p.grad.clone())

        depth_loss.backward(retain_graph=True)
        for p in model.encoder.parameters():
            enc_g2.append(p.grad.clone())

        normal_loss.backward(retain_graph=True)
        for p in model.encoder.parameters():
            enc_g3.append(p.grad.clone())

        mag1 = model.decoder_segmt.dec_block1.conv1[0].weight.grad.abs().mean()
        mag2 = model.decoder_depth.dec_block1.conv1[0].weight.grad.abs().mean()
        mag3 = model.decoder_normal.dec_block1.conv1[0].weight.grad.abs().mean()
        
        for p in model.decoder_segmt.parameters():
            p.grad *= mag1 / (mag1 + mag2 + mag3)
        for p in model.decoder_depth.parameters():
            p.grad *= mag2 / (mag1 + mag2 + mag3)
        for p in model.decoder_depth.parameters():
            p.grad *= mag3 / (mag1 + mag2 + mag3)
        for i, p in enumerate(model.encoder.parameters()):
            p.grad = mag1 / (mag1 + mag2 + mag3) * enc_g1[i] + mag2 / (mag1 + mag2 + mag3) * enc_g2[i] + mag3 / (mag1 + mag2 + mag3) * enc_g3[i]

        optimizer.step()

    return (segmt_loss + depth_loss + normal_loss).item()

parser = argparse.ArgumentParser(description='Reproduction code of AdaMTNet for NYU (3 tasks)')
parser.add_argument('--input_path', default='./data/nyu')
parser.add_argument('--weights_path', default='./exps/model/adamtnet_nyu3tasks.pth')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--infer_only', action='store_true')
parser.add_argument('--use_adalsp', action='store_true')
parser.add_argument('--cpu', action='store_true')
opt = parser.parse_args()

device = torch.device('cuda' if not opt.cpu else 'cpu')
model = AdaMTNet3tasks(out_features_seg=13, batch_size=4, height=288, width=384)
model.to(device)

print("AdaMTNet for NYU dataset (3 tasks)")
if opt.use_adalsp:
    print("Using adaptive loss-specific weight learning")
else:
    print("Equal weights")
print("===============================================================")
print("===============================================================")

print("Parameter Space: {}".format(count_parameters(model)))

train_data = NYUv2(root_path=opt.input_path, split='train', transforms=True, num_tasks=3)
valid_data = NYUv2(root_path=opt.input_path, split='val', transforms=None, num_tasks=3)

train = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
valid = DataLoader(valid_data, batch_size=4, shuffle=True, num_workers=4)

criterion_segmt = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
criterion_depth = nn.L1Loss()
criteria = [criterion_segmt, criterion_depth]
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_valid_loss = 1e5
train_losses = []
valid_losses = []

if not opt.infer_only:
    for epoch in range(1, opt.epochs + 1):
        start = time.time()
        train_loss = 0.
        valid_loss = 0.

        for i, batch in enumerate(train):
            batch_X, batch_y_segmt, batch_y_depth, batch_y_normal = batch
            if not opt.use_adalsp:
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, batch_y_normal, 
                                    model,
                                    criteria=criteria, optimizer=optimizer,
                                    is_train=True)
            else:
                loss = compute_loss_with_adalsp(batch_X, batch_y_segmt, batch_y_depth, batch_y_normal,
                                                model,
                                                criteria=criteria, optimizer=optimizer,
                                                is_train=True)
                            
            train_loss += loss

        for i, batch in enumerate(valid):
            batch_X, batch_y_segmt, batch_y_depth, batch_y_normal = batch
            if not opt.use_adalsp:
                loss = compute_loss(batch_X, batch_y_segmt, batch_y_depth, batch_y_normal, 
                                    model,
                                    criteria=criteria, optimizer=optimizer,
                                    is_train=False)
            else:
                loss = compute_loss_with_adalsp(batch_X, batch_y_segmt, batch_y_depth, batch_y_normal, 
                                                model,
                                                criteria=criteria, optimizer=optimizer,
                                                is_train=False)
            valid_loss += loss

        train_loss /= len(train.dataset)
        valid_loss /= len(valid.dataset)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        elapsed_time = (time.time() - start) / 60
        print("Epoch {}/{} [{:.1f}min] --- train loss: {:.5f} --- valid loss: {:.5f}".format(
                    epoch, opt.epochs, elapsed_time, train_loss, valid_loss))

        if epoch == 0 or valid_loss < best_valid_loss:
            print("Saving weights...")
            weights = model.state_dict()
            torch.save(weights, opt.weights_path)
            best_valid_loss = valid_loss

test_data = NYUv2(root_path=opt.input_path, split='val', transforms=None, num_tasks=3)
test = DataLoader(test_data, batch_size=1, shuffle=True)

dummy_input = torch.randn(1, 3, 288, 384, dtype=torch.float).to(device)
for _ in range(100):
    _ = model(dummy_input)

model.load_state_dict(torch.load(opt.weights_path, map_location=device))
logger = Logger(num_classes=13, ignore_index=-1)
best_score = 0
best_set = {}

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
elapsed_times = []

model.eval()
with torch.no_grad():
    for i, batch in enumerate(test):
        batch_X, batch_y_segmt, batch_y_depth, batch_y_normal = batch
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y_segmt = batch_y_segmt.to(device, non_blocking=True)
        batch_y_depth = batch_y_depth.to(device, non_blocking=True)
        batch_y_normal = batch_y_normal.to(device, non_blocking=True)

        start.record()
        preds = model(batch_X)
        end.record()

        torch.cuda.synchronize()
        batch_time = start.elapsed_time(end)
        elapsed_times.append(batch_time)

        targets = [batch_y_segmt, batch_y_depth]
        logger.log(preds, targets)

print("Avg inference time: {:.3f}[ms]".format(np.mean(elapsed_times)))
logger.get_scores()
