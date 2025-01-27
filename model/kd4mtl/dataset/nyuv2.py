from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np
import pdb
import torchvision.transforms as transforms
from PIL import Image


class NYUv2(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'
        
        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

        self.scale = [1.0, 1.2, 1.5]

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        segmt = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        # apply data augmentation if required
        if self.augmentation:
            image, segmt, depth, normal = self._random_scale_crop(image, segmt, depth, normal)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                segmt = torch.flip(segmt, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = - normal[0, :, :]

        return image.float(), segmt.float(), depth.float(), normal.float()

    def __len__(self):
        return self.data_len

    def _random_scale_crop(self, image, segmt, depth, normal):
        height, width = image.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(image[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(segmt[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_
