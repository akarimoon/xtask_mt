import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TTDown(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None):
        super().__init__()
        if not mid_features:
            mid_features = out_features
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.double_conv(x))

class TTUp(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None):
        super().__init__()
        if not mid_features:
            mid_features = out_features
        self.double_conv = nn.Sequential(
                    nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.double_conv(self.up(x))

class BaseTaskTransferNet(nn.Module):
    """
    Base Task-Transfer Net
    predict y_a->b from y_a
    no transferred features from y_b decoder
    """
    def __init__(self, in_features, out_features, features=[64, 128, 256]):
        super().__init__()
        self.enc1 = TTDown(in_features=in_features, out_features=features[0])
        self.enc2 = TTDown(in_features=features[0], out_features=features[1])
        self.enc3 = TTDown(in_features=features[1], out_features=features[2])
        self.dec1 = TTUp(in_features=features[2], out_features=features[1])
        self.dec2 = TTUp(in_features=features[1], out_features=features[0])
        self.dec3 = TTUp(in_features=features[0], out_features=features[0])
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features
        
        self.name = "Base Task TransferNet"

    def forward(self, x):
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.final_conv(out)
        if self.out_features > 1:
            out = self.nonlinear(out)
        return out

class BaseTaskTransferNetWithSkipCN(nn.Module):
    """
    Base Task-Transfer Net
    predict y_a->b from y_a
    no transferred features from y_b decoder
    """
    def __init__(self, in_features, out_features, features=[64, 128, 256]):
        super().__init__()
        self.enc1 = TTDown(in_features=in_features, out_features=features[0])
        self.enc2 = TTDown(in_features=features[0], out_features=features[1])
        self.enc3 = TTDown(in_features=features[1], out_features=features[2])
        self.dec1 = TTUp(in_features=features[2], out_features=features[1])
        self.dec2 = TTUp(in_features=features[1] + features[1], out_features=features[0])
        self.dec3 = TTUp(in_features=features[0] + features[0], out_features=features[0])
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features

        self.name = "Base Task TransferNet with skip-connection"

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        out = self.dec1(e3)
        out = self.dec2(torch.cat((out, e2), dim=1))
        out = self.dec3(torch.cat((out, e1), dim=1))
        out = self.final_conv(out)
        if self.out_features > 1:
            out = self.nonlinear(out)
        return out

# ================================
# NOT IN USE BELOW
# ================================

class TaskTransferNet(nn.Module):
    """
    Task-Transfer Net
    predict y_a->b from y_a
    transferred features from before final_conv of the other decoder
    concat before encoding
    """
    def __init__(self, in_features, in_features_cross, out_features):
        super().__init__()
        features = [64, 128, 256]
        self.first_conv = nn.Conv2d(in_features_cross, 32, kernel_size=1, stride=1, bias=False)
        self.enc1 = TTDown(in_features=in_features + 32, out_features=features[0])
        self.enc2 = TTDown(in_features=features[0], out_features=features[1])
        self.enc3 = TTDown(in_features=features[1], out_features=features[2])
        self.dec1 = TTUp(in_features=features[2], out_features=features[1])
        self.dec2 = TTUp(in_features=features[1], out_features=features[0])
        self.dec3 = TTUp(in_features=features[0], out_features=features[0])
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features
# 
    def name(self):
        return "Task TransferNet, concat before encoding"
# 
    def forward(self, x1, x2):
        x2 = self.first_conv(x2)
        out = self.enc1(torch.cat((x1, x2), dim=1))
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.final_conv(out)
        if self.out_features > 1:
            out = self.nonlinear(out)
        return out
# 
class TaskTransferNetWithSkipCN(nn.Module):
    """
    Task-Transfer Net with skip connection
    predict y_a->b from y_a
    transferred features from before final_conv of the other decoder
    concat before encoding
    with skip connection
    """
    def __init__(self, in_features, in_features_cross, out_features):
        super().__init__()
        features = [64, 128, 256]
        self.first_conv = nn.Conv2d(in_features_cross, 32, kernel_size=1, stride=1, bias=False)
        self.enc1 = TTDown(in_features=in_features + 32, out_features=features[0])
        self.enc2 = TTDown(in_features=features[0], out_features=features[1])
        self.enc3 = TTDown(in_features=features[1], out_features=features[2])
        self.dec1 = TTUp(in_features=features[2], out_features=features[1])
        self.dec2 = TTUp(in_features=features[1] + features[1], out_features=features[0])
        self.dec3 = TTUp(in_features=features[0] + features[0], out_features=features[0])
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features
# 
    def name(self):
        return "Task TransferNet with skip connection, concat before encoding"
# 
    def forward(self, x1, x2):
        x2 = self.first_conv(x2)
        e1 = self.enc1(torch.cat((x1, x2), dim=1))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        out = self.dec1(e3)
        out = self.dec2(torch.cat((out, e2), dim=1))
        out = self.dec3(torch.cat((out, e1), dim=1))
        out = self.final_conv(out)
        if self.out_features > 1:
            out = self.nonlinear(out)
        return out
# 
class TaskTransferNetTwoEncoder1(nn.Module):
    """
    Task-Transfer Net with 2 encoders (1)
    predict y_a->b from y_a
    transferred features from before final_conv of the other decoder
    2 encoders for y_a and transferred features
    concat before decoder
    """
    def __init__(self, in_features, in_features_cross, out_features):
        super().__init__()
        features = [64, 128, 256]
        self.enc1 = TTDown(in_features=in_features, out_features=features[0])
        self.enc2 = TTDown(in_features=features[0], out_features=features[1])
        self.enc3 = TTDown(in_features=features[1], out_features=features[1])
# 
        self.t_enc1 = TTDown(in_features=in_features_cross, out_features=features[1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 
        self.dec1 = TTUp(in_features=features[1] * 2, out_features=features[1])
        self.dec2 = TTUp(in_features=features[1], out_features=features[0])
        self.dec3 = TTUp(in_features=features[0], out_features=features[0])
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features
# 
    def name(self):
        return "Task TransferNet, 2 encoders and concat before decoder"
# 
    def forward(self, x1, x2):
        enc = self.enc1(x1)
        enc = self.enc2(enc)
        enc = self.enc3(enc)
        t_enc = self.t_enc1(x2)
        t_enc = self.pool(self.pool(t_enc))
        out = self.dec1(torch.cat((enc, t_enc), dim=1))
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.final_conv(out)
        if self.out_features > 1:
            out = self.nonlinear(out)
        return out
# 
class BalanceUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)
# 
    def forward(self, x1, x2):
        b = self.conv1(torch.cat((x1, x2), dim=1))
        out = self.conv2(torch.cat((torch.mul(b, x1), torch.mul(1 - b, x2)), dim=1))
        return out
# 
class TaskTransferNetTwoEncoder2(nn.Module):
    """
    Task-Transfer Net with 2 encoders (2)
    predict y_a->b from y_a
    transferred features from before final_conv of the other decoder
    2 encoders for y_a and transferred features
    attention-based balance module before decoder
    """
    def __init__(self, in_features, in_features_cross, out_features):
        super().__init__()
        features = [64, 128, 256]
        self.enc1 = TTDown(in_features=in_features, out_features=features[0])
        self.enc2 = TTDown(in_features=features[0], out_features=features[1])
        self.enc3 = TTDown(in_features=features[1], out_features=features[1])
# 
        self.t_enc1 = TTDown(in_features=in_features_cross, out_features=features[1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 
        self.b_unit = BalanceUnit(in_features=features[1] * 2, out_features=features[1])
# 
        self.dec1 = TTUp(in_features=features[1], out_features=features[1])
        self.dec2 = TTUp(in_features=features[1], out_features=features[0])
        self.dec3 = TTUp(in_features=features[0], out_features=features[0])
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features
# 
    def name(self):
        return "Task TransferNet, 2 encoders and balance module before decoder"
# 
    def forward(self, x1, x2):
        enc = self.enc1(x1)
        enc = self.enc2(enc)
        enc = self.enc3(enc)
        t_enc = self.t_enc1(x2)
        t_enc = self.pool(self.pool(t_enc))
        out = self.b_unit(enc, t_enc)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.final_conv(out)
        if self.out_features > 1:
            out = self.nonlinear(out)
        return out
