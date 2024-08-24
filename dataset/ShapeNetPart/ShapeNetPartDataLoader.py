import json
import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

import dataset.provider as provider
from dataset.datasset_utils import farthest_point_sample, correct_normal_direction

warnings.filterwarnings('ignore')


def my_collate_fn(batch, is_train):
    batch_point_data = []
    batch_point_normal = []
    batch_point_seg = []
    batch_point_cls = []
    for point_data, point_normal, point_seg, point_cls in batch:
        batch_point_data.append(point_data)
        batch_point_normal.append(point_normal)
        batch_point_seg.append(point_seg)
        batch_point_cls.append(point_cls)
    batch_point_xyz = np.array(batch_point_data)
    batch_point_normal = np.array(batch_point_normal)
    batch_point_seg = np.array(batch_point_seg)
    batch_point_cls = np.array(batch_point_cls)
    # ------------------------------------------------------------------------------------------------------------------
    # Data normalization
    # ------------------------------------------------------------------------------------------------------------------
    if is_train:
        batch_point_xyz = provider.random_scale_point_cloud(batch_point_xyz)
        batch_point_xyz = provider.random_point_dropout(batch_point_xyz)
    batch_point_xyz = provider.normalize_data(batch_point_xyz)
    # ------------------------------------------------------------------------------------------------------------------
    # convert to tensor
    # ------------------------------------------------------------------------------------------------------------------
    batch_point_xyz = torch.from_numpy(batch_point_xyz)
    batch_point_seg = torch.from_numpy(batch_point_seg)
    batch_point_cls = torch.from_numpy(batch_point_cls)
    batch_point_normal = torch.from_numpy(batch_point_normal)
    return batch_point_xyz, batch_point_seg, batch_point_cls, batch_point_normal


class ShapeNetPartDataset(Dataset):
    def __init__(self, args, split, class_choice=None):
        super().__init__()
        self.root = args.data_path
        self.npoints = args.num_point
        self.catfile = os.path.join(self.root, 'class2file.txt')
        self.cat = {}
        self.uniform = args.use_uniform_sample
        # --------------------------------------------------------------------------------------------------------------
        # Read the class file
        # --------------------------------------------------------------------------------------------------------------
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        if class_choice:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        # --------------------------------------------------------------------------------------------------------------
        # Read the point cloud file
        # --------------------------------------------------------------------------------------------------------------
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'train_val':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % split)
                exit(-1)
            for fn in fns:
                token = os.path.basename(fn)
                self.meta[item].append(os.path.join(dir_point, token))
        # --------------------------------------------------------------------------------------------------------------
        # Save the point cloud file storage path
        # --------------------------------------------------------------------------------------------------------------
        self.data_path = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.data_path.append((item, fn))
        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        global point_set, class_name, rotated_point_cloud_data
        if index in self.cache:
            new_point_cloud_data, new_point_cloud_normal, point_cloud_seg, point_cloud_cls = self.cache[index]
        else:
            # ----------------------------------------------------------------------------------------------------------
            # Read point cloud data, point cloud categories, and point cloud segmentation labels
            # ----------------------------------------------------------------------------------------------------------
            class_name, file_path = self.data_path[index][0], self.data_path[index][1]
            point_set = np.loadtxt(str(file_path)).astype(np.float32)
            point_cloud_data = point_set[:, 0:6]
            point_cloud_cls = np.array([self.classes[class_name]]).astype(np.int32)
            point_cloud_seg = point_set[:, -1].astype(np.int32)
            # --------------------------------------------------------------------------------------------------------------
            # Random sampling or farthest point sampling
            # --------------------------------------------------------------------------------------------------------------
            if self.uniform:
                choice = farthest_point_sample(point_set, self.npoints)
                choice = np.hstack(choice)
            else:
                choice = np.random.choice(len(point_cloud_data), self.npoints, replace=True)
            point_cloud_data = point_cloud_data[choice, :]
            point_cloud_seg = point_cloud_seg[choice]
            # ----------------------------------------------------------------------------------------------------------
            # Orient normal vectors
            # ----------------------------------------------------------------------------------------------------------
            new_point_cloud_data, new_point_cloud_normal, differences = correct_normal_direction(point_cloud_data)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (new_point_cloud_data, new_point_cloud_normal, point_cloud_seg, point_cloud_cls)
        return new_point_cloud_data, new_point_cloud_normal, point_cloud_seg, point_cloud_cls

    def __len__(self):
        return len(self.data_path)
