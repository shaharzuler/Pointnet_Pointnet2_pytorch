import os

import numpy as np
import torch

from Pointnet_Pointnet2_pytorch.data_utils.PartCustomDataset.PartCsutomDataset import PartCustomDataset
from Pointnet_Pointnet2_pytorch.models.pointnet_util import pc_normalize

npoints = 4096
root = 'data/custom_partseg_data/'
normal_channel = True
num_augs = 10
TRAIN_DATASET = PartCustomDataset(root=root, npoints=npoints, split='train', normal_channel=normal_channel, is_train=True)
trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=4)  # , pin_memory=True, drop_last=True,


def write_sample(aug, pointset, seg, i):
    name = str(i) + "_" + str(aug)
    filenames.append(name + "\n")
    point_cloud_path = os.path.join(root, "Points", name + ".txt")
    label_path = os.path.join(root, "Labels", name + ".txt")
    np.savetxt(point_cloud_path, pointset)
    np.savetxt(label_path, seg)


filenames = []
for i, data in enumerate(trainDataLoader):
    point_set, seg = data
    point_set, seg = point_set[0, :, :].cpu().numpy(), seg[0, :].cpu().numpy()
    if point_set.shape[1] == 7:  # case argb format
        point_set = np.delete(point_set, 3, axis=1)  # remove a column
    if not normal_channel:
        point_set = point_set[:, 0:3]
    else:
        point_set = point_set[:, 0:6]
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    point_set[:, 3:] /= 255

    for aug in range(num_augs):
        # resample (randomly):
        downsample_ind = np.random.choice(len(seg), npoints, replace=True)
        point_set_aug = point_set[downsample_ind, :]
        seg_aug = seg[downsample_ind]

        point_set_aug = TRAIN_DATASET.augment(point_set_aug)

        write_sample(aug, point_set_aug, seg_aug, i)

with open(root + "train_preprocessed.txt", "w+") as f:
    f.writelines(filenames)
