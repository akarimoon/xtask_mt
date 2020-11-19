import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)

class TTDown(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, batch_norm=False):
        super().__init__()
        if not mid_features:
            mid_features = out_features
        self.double_conv = nn.Sequential(
            SingleConvBlock(in_features, mid_features, batch_norm=batch_norm),
            SingleConvBlock(mid_features, out_features, batch_norm=batch_norm),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.double_conv(x))

class TTUp(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, batch_norm=False):
        super().__init__()
        if not mid_features:
            mid_features = out_features
        self.double_conv = nn.Sequential(
            SingleConvBlock(in_features, mid_features, batch_norm=batch_norm),
            SingleConvBlock(mid_features, out_features, batch_norm=batch_norm),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.double_conv(self.up(x))

class TaskTransferNet(nn.Module):
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
        if self.out_features == 1:
            x = self.nonlinear(x)
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.final_conv(out)
        return out

class TaskTransferNetWithSkipCN(nn.Module):
    """
    Base Task-Transfer Net
    predict y_a->b from y_a
    no transferred features from y_b decoder
    """
    def __init__(self, in_features, out_features, features=[64, 128, 256], batch_norm=False):
        super().__init__()
        self.enc1 = TTDown(in_features=in_features, out_features=features[0], batch_norm=batch_norm)
        self.enc2 = TTDown(in_features=features[0], out_features=features[1], batch_norm=batch_norm)
        self.enc3 = TTDown(in_features=features[1], out_features=features[2], batch_norm=batch_norm)
        self.dec1 = TTUp(in_features=features[2], out_features=features[1], batch_norm=batch_norm)
        self.dec2 = TTUp(in_features=features[1] + features[1], out_features=features[0], batch_norm=batch_norm)
        self.dec3 = TTUp(in_features=features[0] + features[0], out_features=features[0], batch_norm=batch_norm)
        self.final_conv = nn.Conv2d(features[0], out_features, kernel_size=1, stride=1, bias=False)
        self.nonlinear = nn.LogSoftmax(dim=1)
        self.out_features = out_features

        self.name = "Base Task TransferNet with skip-connection"

    def forward(self, x):
        if self.out_features == 1:
            x = self.nonlinear(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        out = self.dec1(e3)
        out = self.dec2(torch.cat((out, e2), dim=1))
        out = self.dec3(torch.cat((out, e1), dim=1))
        out = self.final_conv(out)
        return out
