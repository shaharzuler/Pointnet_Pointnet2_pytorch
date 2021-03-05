import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from Pointnet_Pointnet2_pytorch.models.pointnet_util import pc_normalize


class PartCustomDataset(Dataset):
    def __init__(self, root='./data/custom_partseg_data', npoints=4096, split='train', class_choice=None, normal_channel=False, is_train=True, minimal_preprocess=True):
        self.npoints = npoints
        self.root = root
        self.point_clouds_dir = os.path.join(self.root, "Points")
        self.seg_dir = os.path.join(self.root, "Labels")
        self.normal_channel = normal_channel
        if minimal_preprocess and is_train:
            split_ = "train_preprocessed"
        try:
            with open(os.path.join(self.root, split_ + '.txt')) as f:
                self.ids = f.read().splitlines()
        except:
            with open(os.path.join(self.root, split + '.txt')) as f:
                self.ids = f.read().splitlines()
        self.labelweights = np.array([0.2, 0.8])  # np.array([1., 1.]) #todo calculate from train set
        self.is_train = is_train
        self.minimal_preprocess = minimal_preprocess

    def __getitem__(self, index):
        point_cloud_path = os.path.join(self.point_clouds_dir, str(self.ids[index]) + ".txt")
        point_set = np.loadtxt(point_cloud_path).astype(np.float32)
        if not (self.minimal_preprocess):
            if point_set.shape[1] == 7:  # case xyzargb format
                point_set = np.delete(point_set, 3, axis=1)
            if not self.normal_channel:
                point_set = point_set[:, 0:3]
            else:
                point_set = point_set[:, 0:6]

        seg_path = os.path.join(self.seg_dir, str(self.ids[index]) + ".txt")
        seg = np.loadtxt(seg_path).astype(np.int32)
        reduce_label = 0 if 0 in seg else 1
        seg -= reduce_label

        seg[seg == 2] = 1  # hard coded for reducing 2 classes into 1

        if not (self.minimal_preprocess):
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            point_set[:, 3:] /= 255

            # resample (randomly):
            downsample_ind = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[downsample_ind, :]
            seg = seg[downsample_ind]

            if self.is_train:
                point_set = self.augment(point_set)

        return torch.tensor(point_set), torch.tensor(seg)

    def __len__(self):
        return len(self.ids)

    def augment(self, point_set):
        point_set = self.random_mirror(point_set, axis=0)
        point_set = self.random_mirror(point_set, axis=1)
        return point_set

    def random_mirror(self, point_set, axis):
        mirror_flag = random.randint(0, 1)
        if mirror_flag:
            point_set[:, axis] *= -1  # todo verify that is the correct orientation with the real dataset
        return point_set
