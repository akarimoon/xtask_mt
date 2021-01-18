import numpy as numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
    
class DecoderConvBlock(nn.Module):
    def __init__(self, in_features, out_features, size=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout2d()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.W = nn.Parameter(torch.Tensor(torch.rand(size=size)))
    
    def forward(self, x, e):
        x = torch.cat([self.up(x), e], dim=1)
        l1 = self.conv1(x)
        l2 = self.conv2(l1)
        attn = (self.W[:l1.shape[0]] * l1) * l2
        out = self.dropout(torch.cat([l2, attn], dim=1))
        return out

class EncoderSequential(nn.Module):
    def __init__(self, in_features, out_features, mid_features=[64, 128, 256, 512]):
        super().__init__()
        self.enc_block1 = EncoderConvBlock(in_features=in_features, 
                                       out_features=mid_features[0])
        self.enc_block2 = EncoderConvBlock(in_features=mid_features[0], 
                                       out_features=mid_features[1])
        self.enc_block3 = EncoderConvBlock(in_features=mid_features[1] + mid_features[0],
                                       out_features=mid_features[2])
        self.enc_block4 = EncoderConvBlock(in_features=mid_features[2] + mid_features[1] + mid_features[0], 
                                       out_features=mid_features[3])
        self.bottleneck = nn.Conv2d(mid_features[3] + mid_features[2] + mid_features[1] + mid_features[0],
                                    out_features, kernel_size=1)
        self.dropout = nn.Dropout2d()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc_block1(x)
        out1 = self.pool(self.dropout(e1))
        e2 = self.enc_block2(out1)
        out2 = self.pool(self.dropout(e2))
        e3 = self.enc_block3(torch.cat([out2, self.pool(e1)], dim=1))
        out3 = self.pool(self.dropout(e3))
        e4 = self.enc_block4(torch.cat([out3, self.pool(e2), self.pool(self.pool(1))], dim=1))
        out4 = self.pool(self.dropout(e4))
        out = self.bottleneck(torch.cat([out4, self.pool(e3), self.pool(self.pool(e2)), self.pool(self.pool(self.pool(e1)))], dim=1))
        return out

class DecoderSequential(nn.Module):
    def __init__(self, in_features, out_features, mid_features=[64, 128, 256, 512], batch_size=8, height=128, width=256):
        super().__init__()
        self.dec_block1 = DecoderConvBlock(in_features=in_features + mid_features[3], 
                                           out_features=mid_features[3] // 2, 
                                           size=(batch_size, mid_features[3] // 2, height // 8, width // 8))
        self.dec_block2 = DecoderConvBlock(in_features=mid_features[3] + mid_features[2], 
                                           out_features=mid_features[2] // 2,
                                           size=(batch_size, mid_features[2] // 2, height // 4, width // 4))
        self.dec_block3 = DecoderConvBlock(in_features=mid_features[2] + mid_features[1], 
                                           out_features=mid_features[1] // 2,
                                           size=(batch_size, mid_features[1] // 2, height // 2, width // 2))
        self.dec_block4 = DecoderConvBlock(in_features=mid_features[1] + mid_features[0], 
                                           out_features=mid_features[0] // 2,
                                           size=(batch_size, mid_features[0] // 2, height, width))
        self.final_conv = nn.Conv2d(mid_features[0], out_features, kernel_size=1, stride=1, bias=False)

class AdaMTNet(nn.Module):
    def __init__(self, in_features=3, out_features_seg=7, out_features_dep=1,
                 bn_features=1024, mid_features=[64, 128, 256, 512],
                 batch_size=8, height=128, width=256):
        super(AdaMTNet, self).__init__()
        self.encoder = EncoderSequential(in_features, bn_features)
        self.decoder_segmt = DecoderSequential(bn_features, out_features_seg, batch_size=batch_size, height=height, width=width)
        self.decoder_depth = DecoderSequential(bn_features, out_features_dep, batch_size=batch_size, height=height, width=width)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e1 = self.encoder.enc_block1(x)
        out1 = self.pool(self.dropout(e1))
        e2 = self.encoder.enc_block2(out1)
        out2 = self.pool(self.dropout(e2))
        e3 = self.encoder.enc_block3(torch.cat([
                                        out2,
                                        self.pool(self.pool(e1))
                                    ], dim=1))
        out3 = self.pool(self.dropout(e3))
        e4 = self.encoder.enc_block4(torch.cat([
                                        out3, 
                                        self.pool(self.pool(e2)), 
                                        self.pool(self.pool(self.pool(e1)))
                                    ], dim=1))
        out4 = self.pool(self.dropout(e4))
        out = self.encoder.bottleneck(torch.cat([
                                        out4, 
                                        self.pool(self.pool(e3)), 
                                        self.pool(self.pool(self.pool(e2))), 
                                        self.pool(self.pool(self.pool(self.pool(e1))))
                                    ], dim=1))

        seg4 = self.decoder_segmt.dec_block1(out, e4)
        seg3 = self.decoder_segmt.dec_block2(seg4, e3)
        seg2 = self.decoder_segmt.dec_block3(seg3, e2)
        seg1 = self.decoder_segmt.dec_block4(seg2, e1)
        seg_out = self.decoder_segmt.final_conv(seg1)

        dep4 = self.decoder_depth.dec_block1(out, e4)
        dep3 = self.decoder_depth.dec_block2(dep4, e3)
        dep2 = self.decoder_depth.dec_block3(dep3, e2)
        dep1 = self.decoder_depth.dec_block4(dep2, e1)
        dep_out = self.decoder_depth.final_conv(dep1)

        return seg_out, dep_out

class AdaMTNet3tasks(nn.Module):
    def __init__(self, in_features=3, out_features_seg=7, out_features_dep=1, out_features_nor=1,
                 bn_features=1024, mid_features=[64, 128, 256, 512],
                 batch_size=8, height=128, width=256):
        super(AdaMTNet3tasks, self).__init__()
        self.encoder = EncoderSequential(in_features, bn_features)
        self.decoder_segmt = DecoderSequential(bn_features, out_features_seg, batch_size=batch_size, height=height, width=width)
        self.decoder_depth = DecoderSequential(bn_features, out_features_dep, batch_size=batch_size, height=height, width=width)
        self.decoder_normal = DecoderSequential(bn_features, out_features_nor, batch_size=batch_size, height=height, width=width)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e1 = self.encoder.enc_block1(x)
        out1 = self.pool(self.dropout(e1))
        e2 = self.encoder.enc_block2(out1)
        out2 = self.pool(self.dropout(e2))
        e3 = self.encoder.enc_block3(torch.cat([
                                        out2,
                                        self.pool(self.pool(e1))
                                    ], dim=1))
        out3 = self.pool(self.dropout(e3))
        e4 = self.encoder.enc_block4(torch.cat([
                                        out3, 
                                        self.pool(self.pool(e2)), 
                                        self.pool(self.pool(self.pool(e1)))
                                    ], dim=1))
        out4 = self.pool(self.dropout(e4))
        out = self.encoder.bottleneck(torch.cat([
                                        out4, 
                                        self.pool(self.pool(e3)), 
                                        self.pool(self.pool(self.pool(e2))), 
                                        self.pool(self.pool(self.pool(self.pool(e1))))
                                    ], dim=1))

        seg4 = self.decoder_segmt.dec_block1(out, e4)
        seg3 = self.decoder_segmt.dec_block2(seg4, e3)
        seg2 = self.decoder_segmt.dec_block3(seg3, e2)
        seg1 = self.decoder_segmt.dec_block4(seg2, e1)
        seg_out = self.decoder_segmt.final_conv(seg1)

        dep4 = self.decoder_depth.dec_block1(out, e4)
        dep3 = self.decoder_depth.dec_block2(dep4, e3)
        dep2 = self.decoder_depth.dec_block3(dep3, e2)
        dep1 = self.decoder_depth.dec_block4(dep2, e1)
        dep_out = self.decoder_depth.final_conv(dep1)

        nor4 = self.decoder_normal.dec_block1(out, e4)
        nor3 = self.decoder_normal.dec_block2(nor4, e3)
        nor2 = self.decoder_normal.dec_block3(nor3, e2)
        nor1 = self.decoder_normal.dec_block4(nor2, e1)
        nor_out = self.decoder_normal.final_conv(nor1)
        nor_out = nor_out / torch.norm(nor_out, p=2, dim=1, keepdim=True)

        return seg_out, dep_out, nor_out

