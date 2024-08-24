import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

import dataset.provider as provider
from dataset.datasset_utils import correct_normal_direction

warnings.filterwarnings('ignore')


def my_collate_fn(batch, train=False):
    batch_point_data = []
    batch_point_seg = []
    batch_point_normal = []
    batch_point_weight = []
    for point_data, point_normal, point_seg, weight in batch:
        batch_point_data.append(point_data)
        batch_point_normal.append(point_normal)
        batch_point_seg.append(point_seg)
        batch_point_weight.append(weight)
    batch_point_data = np.array(batch_point_data)
    batch_point_normal = np.array(batch_point_normal)
    batch_point_seg = np.array(batch_point_seg)
    batch_point_weight = np.array(batch_point_weight)
    # --------------------------------------------------------------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------------------------------------------------------------
    if train:
        batch_point_data = provider.random_point_dropout(batch_point_data)
        batch_point_data[:, :, 0:3] = provider.random_scale_point_cloud(batch_point_data[:, :, 0:3])
        batch_point_data[:, :, 0:3] = provider.shift_point_cloud(batch_point_data[:, :, 0:3])
        batch_point_data[:, :, 0:3] = provider.rotate_point_cloud_z(batch_point_data[:, :, :3])
    # ------------------------------------------------------------------------------------------------------------------
    # Data normalization
    # ------------------------------------------------------------------------------------------------------------------
    batch_point_data[:, :, 0:3] = provider.normalize_data(batch_point_data[:, :, 0:3])
    # ------------------------------------------------------------------------------------------------------------------
    # Convert to tensor
    # ------------------------------------------------------------------------------------------------------------------
    batch_point_data = torch.from_numpy(batch_point_data).type(torch.FloatTensor)
    batch_point_normal = torch.from_numpy(batch_point_normal)
    batch_point_seg = torch.from_numpy(batch_point_seg)
    batch_point_weight = torch.from_numpy(batch_point_weight)
    return batch_point_data, batch_point_normal, batch_point_seg, batch_point_weight


class S3DISDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.data_root = args.data_path
        self.num_point = args.num_point
        self.block_size = args.block_size
        self.test_area = args.test_area
        rooms = sorted(os.listdir(self.data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(self.test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(self.test_area) in room]
        # --------------------------------------------------------------------------------------------------------------
        # Read all scene data
        # --------------------------------------------------------------------------------------------------------------
        self.room_points, self.room_normals, self.room_labels = [], [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        self.class_weights = np.zeros(13)
        for room_name in rooms_split:
            room_path = os.path.join(self.data_root, room_name)
            room_data = np.load(room_path)
            points, normals, labels = room_data[:, 0:6], room_data[:, 6:9], room_data[:, 9]
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the histogram of the 13 point cloud categories in the current scene.
            # ----------------------------------------------------------------------------------------------------------
            class_number, _ = np.histogram(labels, range(14))
            self.class_weights += class_number
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the point with the minimum x-coordinate and the point with the maximum x-coordinate
            # in the current scene.
            # ----------------------------------------------------------------------------------------------------------
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_normals.append(normals), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        # --------------------------------------------------------------------------------------------------------------
        # Determine the number of points for each of the 13 point cloud categories in the entire dataset.
        # Then, divide by the total number of points to get the proportion of each category in the dataset.
        # The fewer the number of point clouds in a category, the greater the weight of that category's point clouds.
        # Using `np.pow` helps to prevent the weight from becoming too large; for example, `4 ** (1/2) = 2`.
        # --------------------------------------------------------------------------------------------------------------
        self.class_weights = self.class_weights.astype(np.float32)
        self.class_weights = self.class_weights / np.sum(self.class_weights)
        self.class_weights = np.power(np.amax(self.class_weights) / self.class_weights, 1 / 3.0)
        self.class_weights[self.class_weights == np.inf] = 1
        # --------------------------------------------------------------------------------------------------------------
        # Calculate the proportion of each room relative to the total number of rooms, denoted as `sample_prob`.
        # For example, if the proportions are `[0.2, 0.4, 0.4]`
        # The total dataset contains `np.sum(num_point_all) -> 4096 * 10` points. Since 4096 points need to be cropped
        # each time, `num_iter = 10` iterations are required for cropping.
        # Then [0.2, 0.4, 0.4] * 10 = [2, 4, 4], which means the first room needs to be cut twice, and the second
        # and the third rooms need to be cut four times.
        # --------------------------------------------------------------------------------------------------------------
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) / self.num_point)
        room_ids = []
        for index in range(len(rooms_split)):
            room_ids.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_ids = np.array(room_ids)

    def __getitem__(self, idx, vis=False):
        room_idx = self.room_ids[idx]
        points_room = self.room_points[room_idx]
        points_normals = self.room_normals[room_idx]
        points_labels = self.room_labels[room_idx]
        points_number = points_room.shape[0]
        while True:
            center = points_room[np.random.choice(points_number)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_ids = np.where((points_room[:, 0] >= block_min[0]) &
                                 (points_room[:, 0] <= block_max[0]) &
                                 (points_room[:, 1] >= block_min[1]) &
                                 (points_room[:, 1] <= block_max[1]))[0]
            if point_ids.size > 1024:
                break
        if point_ids.size >= self.num_point:
            selected_point_ids = np.random.choice(point_ids, self.num_point, replace=False)
        else:
            selected_point_ids = np.random.choice(point_ids, self.num_point, replace=True)
        selected_points = points_room[selected_point_ids, :]
        selected_normals = points_normals[selected_point_ids, :]
        selected_labels = points_labels[selected_point_ids]
        selected_weights = self.class_weights[selected_labels.astype(np.uint8)]
        # --------------------------------------------------------------------------------------------------------------
        # Orientation of Normal Vector
        # --------------------------------------------------------------------------------------------------------------
        _, selected_normals, difference = correct_normal_direction(np.concatenate((selected_points[:, :3],
                                                                                   selected_normals), axis=1))
        current_points = np.zeros((self.num_point, 9))
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        return current_points, selected_normals, selected_labels, selected_weights

    def __len__(self):
        return len(self.room_ids)


class S3DISDatasetWholeScene(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.data_root = args.data_path
        self.num_point = args.num_point
        self.split = split
        self.test_area = args.test_area
        self.stride = args.stride
        self.block_size = args.block_size
        self.padding = args.padding
        # --------------------------------------------------------------------------------------------------------------
        # Reading point cloud data to obtain point cloud scene data, normal vectors, and labels.
        # --------------------------------------------------------------------------------------------------------------
        self.file_list = [d for d in os.listdir(self.data_root) if 'Area_{}'.format(self.test_area) in d]
        self.scene_point_list = []
        self.scene_normal_list = []
        self.scene_lable_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(self.data_root + '/' + file)
            self.scene_point_list.append(data[:, :6])
            self.scene_normal_list.append(data[:, 6:9])
            self.scene_lable_list.append(data[:, 9])
            coord_min, coord_max = np.amin(data[:, :3], axis=0)[:3], np.amax(data[:, :3], axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        # --------------------------------------------------------------------------------------------------------------
        # Counting a histogram of 13 point cloud categories across all scenes.
        # --------------------------------------------------------------------------------------------------------------
        self.scene_points_num = []
        self.class_weights = np.zeros(13)
        for seg in self.scene_lable_list:
            class_number, _ = np.histogram(seg, range(14))
            self.class_weights += class_number
            self.scene_points_num.append(seg.shape[0])
        # --------------------------------------------------------------------------------------------------------------
        # Obtain the number of points for each of the 13 point cloud categories across the entire dataset, and then
        # divide it by the total number to get the proportion of each category's points in the entire dataset.
        # The fewer the number of point clouds in a certain category, the greater the weight of that category's point
        # clouds. Using np.pow is to prevent the weight from being too large, for example, (4 ** 1/2 = 2).
        # --------------------------------------------------------------------------------------------------------------
        self.class_weights = self.class_weights.astype(np.float32)
        self.class_weights = self.class_weights / np.sum(self.class_weights)
        self.class_weights = np.power(np.amax(self.class_weights) / (self.class_weights), 1 / 3.0)
        self.class_weights[self.class_weights == np.inf] = 1

    def __getitem__(self, index):
        points = self.scene_point_list[index]
        normals = self.scene_normal_list[index]
        labels = self.scene_lable_list[index]
        coord_min = self.room_coord_min[index]
        coord_max = self.room_coord_max[index]
        # --------------------------------------------------------------------------------------------------------------
        # (coord_max - coord_min) calculates the length, subtracting self.block_size / self.stride indicates how many
        # steps can be taken, and adding 1 represents including the first block.
        # --------------------------------------------------------------------------------------------------------------
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, normal_room, label_room = np.array([]), np.array([]), np.array([])
        sample_weight, index_room = np.array([]), np.array([])
        for index_y in range(0, grid_x):
            for index_x in range(0, grid_y):
                # ------------------------------------------------------------------------------------------------------
                # Calculate the starting XY coordinates for each crop.
                # The starting point plus the block size may exceed the edge, so when the end_point is at the edge,
                # the starting point should be set at the position of end_point minus the block size.
                # ------------------------------------------------------------------------------------------------------
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_index = np.where((points[:, 0] >= s_x - self.padding) &
                                       (points[:, 0] <= e_x + self.padding) &
                                       (points[:, 1] >= s_y - self.padding) &
                                       (points[:, 1] <= e_y + self.padding))[0]
                if point_index.size == 0:
                    continue
                # ------------------------------------------------------------------------------------------------------
                # For the point cloud with block_size, each time input predicts self.num_point points.
                # ------------------------------------------------------------------------------------------------------
                num_batch = int(np.ceil(point_index.size / self.num_point))
                # ------------------------------------------------------------------------------------------------------
                # 7 /4 = 2， 2 * 4 = 8， 8 -7 =1.
                # The original point cloud has 7 points, but now we need 8 points. We need to sample one more point
                # to make up two point cloud targets, each with 4 points.
                # ------------------------------------------------------------------------------------------------------
                point_size = int(num_batch * self.num_point)
                point_index_repeat = np.random.choice(point_index, point_size - point_index.size, replace=False)
                point_index = np.concatenate((point_index, point_index_repeat))
                data_batch = points[point_index, :]
                normal_batch = normals[point_index, :]
                # ------------------------------------------------------------------------------------------------------
                # Orientation of normal vector
                # ------------------------------------------------------------------------------------------------------
                data_normal_batch = np.concatenate((data_batch[:, :3], normal_batch), axis=1)
                # data_view, normal_batch, difference = correct_normal_direction(data_normal_batch)
                # ------------------------------------------------------------------------------------------------------
                # Normalize the coordinates and colors of the point cloud.
                # ------------------------------------------------------------------------------------------------------
                normalized_xyz = np.zeros((point_size, 3))
                normalized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normalized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normalized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                # ------------------------------------------------------------------------------------------------------
                # The first 6 dimensions are xyz coordinates plus color, and the 6th to 9th dimensions are the
                # normalized coordinates.
                # ------------------------------------------------------------------------------------------------------
                data_batch = np.concatenate((data_batch, normalized_xyz), axis=1)
                label_batch = labels[point_index].astype(int)
                batch_weight = self.class_weights[label_batch]
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                normal_room = np.vstack([normal_room, normal_batch]) if normal_room.size else normal_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if sample_weight.size else batch_weight
                index_room = np.hstack([index_room, point_index]) if index_room.size else point_index
        data_room = data_room.reshape((-1, self.num_point, data_room.shape[1]))
        normal_room = normal_room.reshape((-1, self.num_point, normal_room.shape[1]))
        label_room = label_room.reshape((-1, self.num_point))
        sample_weight = sample_weight.reshape((-1, self.num_point))
        index_room = index_room.reshape((-1, self.num_point))
        # ------------------------------------------------------------------------------------------------------------------
        # Data normalization
        # ------------------------------------------------------------------------------------------------------------------
        data_room[:, :, 0:3] = provider.normalize_data(data_room[:, :, 0:3])
        # ------------------------------------------------------------------------------------------------------------------
        # Convert to tensor
        # ------------------------------------------------------------------------------------------------------------------
        batch_point_data = torch.from_numpy(data_room).type(torch.FloatTensor)
        batch_point_normal = torch.from_numpy(normal_room)
        batch_point_seg = torch.from_numpy(label_room)
        batch_point_weight = torch.from_numpy(sample_weight)
        batch_point_index = torch.from_numpy(index_room)
        return batch_point_data, batch_point_normal, batch_point_seg, batch_point_weight, batch_point_index

    def __len__(self):
        return len(self.scene_point_list)
