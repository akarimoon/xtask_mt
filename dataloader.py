import os
import cv2
import numpy as np
import fnmatch
import h5py
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF

class NYUv2(Dataset):
    """
    From https://github.com/lorenmt/mtan
    """
    def __init__(self, root_path, split='', transforms=None):
        self.transforms = transforms
        self.data_path = os.path.join(root_path, split)
        
        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))

        # apply data augmentation if required
        if self.transforms:
            # image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
            # if torch.rand(1) < 0.5:
            #     image = torch.flip(image, dims=[2])
            #     semantic = torch.flip(semantic, dims=[1])
            #     depth = torch.flip(depth, dims=[2])
            #     normal = torch.flip(normal, dims=[2])
            #     normal[0, :, :] = - normal[0, :, :]
            pass

        return image.float(), semantic.long(), depth.float()

    def __len__(self):
        return self.data_len

class NYUDataset(Dataset):
    def __init__(self, root_path, height, width, split='', transform=None):
        data_path = os.path.join(root_path, 'nyu_depth_v2_labeled.mat')
        splits_path = os.path.join(root_path, 'splits.mat')
        idxs = io.loadmat(splits_path)[split + 'Ndxs']

        with h5py.File(data_path, 'r') as data:
            self.images = data["images"][idxs - 1]
            self.segmts = data["labels"][idxs - 1]
            self.depths = data["depths"][idxs - 1]
        self.height = height
        self.width = width

        if transform is None:
            transform = []
        self.random_crop = True if "random_crop" in transform else False
        self.random_flip = True if "random_flip" in transform else False
        self.random_flip_prob = 0.5

        self.class_map = {
            1: 11, 2: 4, 3: 5, 4: 0, 5: 3,
            6: 8, 7: 9, 8: 11, 9: 12, 10: 5,
            11: 7, 12: 5, 13: 12, 14: 9, 15: 5,
            16: 12, 17: 5, 18: 6, 19: 6, 20: 4,
            21: 6, 22: 2, 23: 1, 24: 5, 25: 10, 
            26: 6, 27: 6, 28: 6, 29: 7, 30: 6,
            31: 6, 32: 5, 33: 6, 34: 6, 35: 6,
            36: 6, 37: 6, 38: 6, 39: 5, 40: 6
        }

    def __getitem__(self, index):
        inputs = {}
        inputs["image"] = self.images[index]
        inputs["segmt"] = self.encode_segmt(self.segmts[index])
        inputs["depth"] = np.array(self.depths[index]).astype(np.float32)
        
        self._transform(inputs)

        image = inputs["image"]
        semgt = inputs["semgt"]
        depth = inputs["depth"]
        
        return image, segmt, depth

    def _transform(self, inputs):
        resize = transforms.Resize(size=(self.height, self.width), interpolation=cv2.INTER_NEAREST)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # array -> PIL
        for k in list(inputs):
            print(k)
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
    
    def encode_segmt(self, mask):
        for _validc in range(1, 41):
            mask[mask == _validc] = self.class_map[_validc]
        
        return mask

    def __len__(self):
        return len(self.images)

class CityscapesDataset(Dataset):
    """
    Code based on torchvision's cityscapes dataset (link below)
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/cityscapes.html#Cityscapes

    See CityscapesScripts' code for the mapping of the 19 classes to 7 classes
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
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
        """
        Given an index, return
        - original: list of image, target segmentation, target depth with the original size
        - image: preprocessed image
        - segmt: preprocessed segmentation
        - depth: preprocessed depth

        Note that Cityscapes dataset only contains disparity maps instead of depth so it needs to be transformed
        The equation used to transform disparity to depth is https://github.com/mcordts/cityscapesScripts/issues/55

        For Cityscapes, we are predicting inverse depth instead of the actual depth because some pixels have infinity depth (e.g. sky)

        The masks are boolean mask to exclude:
        - mask_segmt: pixels with ignore_index
        - mask_depth: pixels with 0 actual depth (pixels which correct depth was not measured)
        """
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
