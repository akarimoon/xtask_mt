import os
import cv2
import numpy as np
import h5py
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF

class NYUDataset(Dataset):
    def __init__(self, mat_path, transform=None):
        with h5py.File(mat_path, 'r') as data:
            self.images = data["images"][()]
            self.targets = data["depths"][()]
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        y = self.targets[index]
        mask = np.ones(x.shape)
        if self.transform:
            x = x.transpose(1, 2, 0)
            sample = self.transform({"image": x, "depth": y, "mask": mask})
            x = sample["image"]
            y = sample["depth"]
        y_inverse = 1 / y

        return x, y_inverse, index

    def __len__(self):
        return len(self.images)

class NYUDatasetExtended(NYUDataset):
    def __init__(self, mat_path, transform=None, sample_count=1000):
        NYUDataset.__init__(self, mat_path, transform)
        self.sample_count = sample_count

    def create_sparse_mask(self, mask_size=384*384):
        mask = np.zeros(mask_size)
        mask[:self.sample_count] = 1
        np.random.shuffle(mask)
        mask = np.reshape(mask, (384, 384))
        return mask

    def __getitem__(self, index):
        x = self.images[index]
        y = self.targets[index]
        mask = self.create_sparse_mask()
        if self.transform:
            x = x.transpose(1, 2, 0)
            sample = self.transform({"image": x, "depth": y, "mask": mask})
            x = sample["image"]
            y = sample["depth"]
        y_inverse = 1 / y

        return x, y_inverse, mask, index

class NYUDatasetWithOriginal(NYUDataset):
    def __init__(self, mat_path, transform=None):
        NYUDataset.__init__(self, mat_path, transform)

    def __getitem__(self, index):
        x_original = self.images[index]
        y_original = self.targets[index]
        mask = np.ones(x_original.shape)
        if self.transform:
            x_original = x_original.transpose(1, 2, 0)
            sample = self.transform({"image": x_original, "depth": y_original, "mask": mask})
            x = sample["image"]
            y = sample["depth"]

        y_inverse = 1 / y

        return x, x_original, y_inverse, index

class CityscapesDataset(Dataset):
    colors_19 = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colors_19 = dict(zip(range(19), colors_19))

    colors_7 = [
        [68, 1, 84],
        [70, 49, 126],
        [54, 92, 141],
        [39, 127, 142],
        [31, 161, 135],
        [74, 193, 109],
        [160, 218, 57]
    ]

    label_colors_7 = dict(zip(range(7), colors_7))

    def __init__(self, height, width, root_path, num_classes=19, split='', transform=None, ignore_index=250):
        """
        transform should be a list
        """
        if num_classes not in [7, 19]:
            raise ValueError("# of classes must be either 7 or 19 but got {}".format(num_classes))

        self.height = height
        self.width = width
        self.root = root_path
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.segmt_dir = os.path.join(self.root, 'gtFine', split)
        self.depth_dir = os.path.join(self.root, 'disparity', split)
        self.split = split
        self.transform = transform
        self.mode = 'gtFine'
        self.images = []
        self.segmts = []
        self.disps = []

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            segmt_targets_dir = os.path.join(self.segmt_dir, city)
            disp_targets_dir = os.path.join(self.depth_dir, city)
            for file_name in os.listdir(img_dir):
                segmt_target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                   self._get_target_suffix(self.mode, 'semantic'))
                disp_target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                   self._get_target_suffix(self.mode, 'disparity'))
                self.images.append(os.path.join(img_dir, file_name))
                self.segmts.append(os.path.join(segmt_targets_dir, segmt_target_name))
                self.disps.append(os.path.join(disp_targets_dir, disp_target_name))

        self.n_classes = num_classes
        if self.n_classes == 19:
            self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
            self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
            self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        else:
            self.void_classes = [0, 1, 2, 3, 4, 5, 6]
            self.valid_classes = list(range(7, 34)) + [-1]
            self.class_map = {
                7: 0, 8: 0, 9: 0, 10: 0,
                11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1,
                17: 2, 18: 2, 19: 2, 20: 2,
                21: 3, 22: 3,
                23: 4,
                24: 5, 25: 5,
                26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 6, -1: 6
            }
        self.ignore_index = ignore_index

        if transform is None:
            transform = []
        self.random_crop = True if "random_crop" in transform else False
        self.random_flip = True if "random_flip" in transform else False
        self.random_flip_prob = 0.5

    def __getitem__(self, index):
        inputs = {}

        image_org = np.array(Image.open(self.images[index]).convert('RGB'))
        inputs["image"] = image_org

        segmt_org = np.array(Image.open(self.segmts[index]))
        inputs["segmt"] = self.encode_segmt(segmt_org)
        inputs["mask_segmt"] = np.float32(inputs["segmt"] != self.ignore_index)

        disp = np.array(Image.open(self.disps[index])).astype(np.float32)
        inputs["mask_depth"] = np.float32(disp > 0)
        disp[disp > 0] = (disp[disp > 0] - 1 ) / 256
        depth_org = disp.copy()
        depth_org[depth_org > 0] = (0.209313 * 2262.52) / depth_org[depth_org > 0]
        inputs["depth"] = depth_org

        self._transform(inputs)

        image = inputs["image"]
        segmt = inputs["segmt"]
        depth = inputs["depth"]
        mask_segmt = inputs["mask_segmt"]
        mask_depth = inputs["mask_depth"]

        original = [image_org, segmt_org, depth_org]

        return original, image, segmt, depth, mask_segmt, mask_depth

    def _transform(self, inputs):
        """
        input
        - inputs: dict
        """
        resize = transforms.Resize(size=(self.height, self.width), interpolation=cv2.INTER_NEAREST)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # array -> PIL
        for k in list(inputs):
            if k == "depth":
                inputs[k][inputs[k] > 0] = 1 / inputs[k][inputs[k] > 0]
            inputs[k] = TF.to_pil_image(inputs[k])

        # resize
        for k in list(inputs):
            inputs[k] = resize(inputs[k])

        # random transformation
        if self.random_crop:
            i, j, h, w = transforms.RandomCrop.get_params(inputs["image"], output_size=size)
            for k in inputs.keys():
                inputs[k] = TF.crop(inputs[k], i, j, h, w)

        if self.random_flip:
            if np.random.rand() > self.random_flip_prob:
                for k in inputs.keys():
                    inputs[k] = TF.hflip(inputs[k])

        # PIL -> tensor
        for k in list(inputs):
            arr = np.array(inputs[k])
            inputs[k] = TF.to_tensor(arr)

        # normalize
        inputs["image"] = normalize(inputs["image"])
        inputs["segmt"] = inputs["segmt"].squeeze()
        inputs["segmt"] *= 255

        # change tensor type
        for k in list(inputs):
            if k == "segmt":
                inputs[k] = inputs[k].type(torch.LongTensor)
            else:
                inputs[k] = inputs[k].type(torch.FloatTensor)

    def decode_segmt(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            if self.n_classes == 19:
                r[temp == l] = self.label_colors_19[l][0]
                g[temp == l] = self.label_colors_19[l][1]
                b[temp == l] = self.label_colors_19[l][2]
            else:
                r[temp == l] = self.label_colors_7[l][0]
                g[temp == l] = self.label_colors_7[l][1]
                b[temp == l] = self.label_colors_7[l][2]


        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmt(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        
        return mask

    def __len__(self):
        return len(self.images)

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'disparity':
            return 'disparity.png'
        else:
            return '{}_polygons.json'.format(mode)
