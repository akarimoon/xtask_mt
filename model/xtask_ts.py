import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .transfer_net import TaskTransferNet, TaskTransferNetWithSkipCN

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, mid_features=128):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_features, mid_features,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._init_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.up(out)
        return out
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DecoderSequential(nn.Module):
    def __init__(self, enc_features, out_features, mid_features=128):
        super(DecoderSequential, self).__init__()
        self.conv1 = ConvBlock(in_features=enc_features[4],
                               out_features=mid_features)
        self.conv2 = ConvBlock(in_features=mid_features + enc_features[3], 
                               out_features=mid_features)
        self.conv3 = ConvBlock(in_features=mid_features + enc_features[2], 
                               out_features=mid_features)
        self.conv4 = ConvBlock(in_features=mid_features + enc_features[1], 
                               out_features=mid_features)
        self.conv5 = ConvBlock(in_features=mid_features + enc_features[0], 
                               out_features=mid_features)
        self.conv6 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, kernel_size=1, stride=1, bias=False)
        )

        self._init_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class XTaskTSNet(nn.Module):
    def __init__(self, enc_layers, out_features_segmt=19, out_features_depth=1, 
                 decoder_mid_features=128,
                 batch_norm=False, use_pretrain=False, wider_ttnet=False):
        super(XTaskTSNet, self).__init__()

        if enc_layers not in [18, 34, 50, 101, 152]:
            raise ValueError("{} is not a valid number of resnet layers".format(enc_layers))

        self.use_pretrain = use_pretrain
        self.pretrained_encoder = self._load_encoder(enc_layers)
        enc_features = np.array([64, 64, 128, 256, 512])
        if enc_layers > 34:
            enc_features[1:] *= 4

        self.decoder_segmt = DecoderSequential(enc_features=enc_features, 
                                               out_features=out_features_segmt, 
                                               mid_features=decoder_mid_features)
        self.decoder_depth = DecoderSequential(enc_features=enc_features, 
                                               out_features=out_features_depth, 
                                               mid_features=decoder_mid_features)
        self.trans_s2d = TaskTransferNetWithSkipCN(in_features=out_features_segmt, 
                                                   out_features=out_features_depth, 
                                                   batch_norm=batch_norm, wider_ttnet=wider_ttnet)
        self.trans_d2s = TaskTransferNetWithSkipCN(in_features=out_features_depth, 
                                                   out_features=out_features_segmt, 
                                                   batch_norm=batch_norm, wider_ttnet=wider_ttnet)
        self.trans_name = self.trans_s2d.name

    def _load_encoder(self, enc_layers):
        try:
            backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnet' + str(enc_layers), pretrained=self.use_pretrain)
        except:
            backbone = models.resnet34(pretrained=False)
            if self.use_pretrain:
                backbone.load_state_dict(torch.load('./model/resnet34-333f7ec4.pth'))

        pretrained = nn.Module()
        pretrained.layer0 = nn.Sequential(
                                backbone.conv1,
                                backbone.bn1,
                                backbone.relu                    
                            )
        pretrained.layer1 = nn.Sequential(
                                backbone.maxpool,
                                backbone.layer1
                            )
        pretrained.layer2 = backbone.layer2
        pretrained.layer3 = backbone.layer3
        pretrained.layer4 = backbone.layer4
        return pretrained

    def forward(self, x, direct_only=False):
        enc0 = self.pretrained_encoder.layer0(x)
        enc1 = self.pretrained_encoder.layer1(enc0)
        enc2 = self.pretrained_encoder.layer2(enc1)
        enc3 = self.pretrained_encoder.layer3(enc2)
        enc4 = self.pretrained_encoder.layer4(enc3)

        seg4 = self.decoder_segmt.conv1(enc4)
        seg3 = self.decoder_segmt.conv2(torch.cat((seg4, enc3), dim=1))
        seg2 = self.decoder_segmt.conv3(torch.cat((seg3, enc2), dim=1))
        seg1 = self.decoder_segmt.conv4(torch.cat((seg2, enc1), dim=1))
        seg0 = self.decoder_segmt.conv5(torch.cat((seg1, enc0), dim=1))
        seg_out = self.decoder_segmt.conv6(seg0)

        dep4 = self.decoder_depth.conv1(enc4)
        dep3 = self.decoder_depth.conv2(torch.cat((dep4, enc3), dim=1))
        dep2 = self.decoder_depth.conv3(torch.cat((dep3, enc2), dim=1))
        dep1 = self.decoder_depth.conv4(torch.cat((dep2, enc1), dim=1))
        dep0 = self.decoder_depth.conv5(torch.cat((dep1, enc0), dim=1))
        dep_out = self.decoder_depth.conv6(dep0)

        if direct_only:
            return seg_out, None, dep_out, None

        dep_tout = self.trans_s2d(seg_out)
        seg_tout = self.trans_d2s(dep_out)

        return seg_out, seg_tout, dep_out, dep_tout

class BaseSTLNet(nn.Module):
    def __init__(self, enc_layers, out_features, decoder_in_features=256, decoder_mid_features=128):
        super(BaseSTLNet, self).__init__()

        if enc_layers not in [18, 34, 50, 101, 152]:
            raise ValueError("{} is not a valid number of resnet layers".format(enc_layers))

        self.pretrained_encoder = self._load_encoder(enc_layers)
        enc_features = np.array([64, 64, 128, 256, 512])
        if enc_layers > 34:
            enc_features[1:] *= 4

        self.decoder = DecoderSequential(enc_features=enc_features, 
                                         out_features=out_features, 
                                         mid_features=decoder_mid_features)

    def _load_encoder(self, enc_layers):
        try:
            backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnet' + str(enc_layers), pretrained=False)
        except:
            backbone = models.resnet34(pretrained=False)

        pretrained = nn.Module()
        pretrained.layer0 = nn.Sequential(
                                backbone.conv1,
                                backbone.bn1,
                                backbone.relu                    
                            )
        pretrained.layer1 = nn.Sequential(
                                backbone.maxpool,
                                backbone.layer1
                            )
        pretrained.layer2 = backbone.layer2
        pretrained.layer3 = backbone.layer3
        pretrained.layer4 = backbone.layer4
        return pretrained

    def forward(self, x):
        enc0 = self.pretrained_encoder.layer0(x)
        enc1 = self.pretrained_encoder.layer1(enc0)
        enc2 = self.pretrained_encoder.layer2(enc1)
        enc3 = self.pretrained_encoder.layer3(enc2)
        enc4 = self.pretrained_encoder.layer4(enc3)

        out4 = self.decoder.conv1(enc4)
        out3 = self.decoder.conv2(torch.cat((out4, enc3), dim=1))
        out2 = self.decoder.conv3(torch.cat((out3, enc2), dim=1))
        out1 = self.decoder.conv4(torch.cat((out2, enc1), dim=1))
        out0 = self.decoder.conv5(torch.cat((out1, enc0), dim=1))
        out = self.decoder.conv6(out0)

        return out
