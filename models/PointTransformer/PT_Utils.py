# -*- coding: utf-8 -*-
# @Time    : 2023-05-14
# @Author  : lab
# @desc    :
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pcd(points):
    """
    Data type conversion，numpy -> point cloud
    @param points:
    @return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def softmax_function(data):
    e_x = torch.exp(data - torch.max(data, dim=-1, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=-1, keepdim=True)


def pc_normalize(pc):
    """
    Normalize point cloud data
    @param pc:
    @return:
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def normalize_pytorch(data):
    """
    N x D np.array data
    """
    data_min = torch.min(data)
    data_max = np.max(data)
    result = (data - data_min) / (data_max - data_min)
    return result


def normalize_pytorch_batch(batch_data):
    """
    B × N x K × C torch data
    """
    B, N, K, C = batch_data.shape
    normal_data = torch.zeros((B, N, K, C))
    for b in range(B):
        data = batch_data[b]
        data_min = torch.min(data, dim=1)[0]
        data_max = torch.max(data, dim=1)[0]
        result = (data - data_min[:, None, :]) / (data_max[:, None, :] - data_min[:, None, :])
        result[torch.isnan(result)] = 1
        normal_data[b] = result
    return normal_data.to(batch_data.device)


def normalize_pytorch_batch_N_N(batch_data):
    """
    B × N x N torch data
    """
    B, N, N = batch_data.shape
    normal_data = torch.zeros((B, N, N), device=batch_data.device)
    for b in range(B):
        data = batch_data[b].to(batch_data.device)
        data_min = torch.min(data, dim=1)[0]
        data_max = torch.max(data, dim=1)[0]
        result = (data - data_min[:, None]) / (1e-9 + data_max[:, None] - data_min[:, None])
        normal_data[b] = result
    return normal_data


def compute_normal_vectors_covariance(knn_xyz, neighbor_num):
    """
    Calculate the normal vector for each point by performing SVD (Singular Value Decomposition) on the covariance matrix.
    @param knn_xyz: The coordinates of KNN: `[batch_size, num_points, k, 3]`.
    @param neighbor_num: the  number of neighborhood points used for fitting the local plane of the normal vector.
    @return:
    """
    knn_xyz = knn_xyz[:, :, :neighbor_num, :]
    center_xyz = torch.mean(knn_xyz, dim=-2)
    neighbor_centered = knn_xyz - center_xyz[:, :, None, :]
    covariance = torch.matmul(neighbor_centered.permute(0, 1, 3, 2), neighbor_centered)
    U, S, V = torch.svd(covariance)
    normal_vectors = V[:, :, :, -1]
    return normal_vectors


def compute_normal_vectors_open3d(xyz, neighbor_num):
    """
    Calculate the normal vectors of the point cloud using Open3D's built-in functions.
    @param xyz: torch, shape(2048, 3)
    @return: normal vector, shape(2048, 3)
    """
    single_pcd = get_pcd(xyz)
    single_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=neighbor_num))
    single_pcd_normal = np.asarray(single_pcd.normals)
    return single_pcd_normal


def normalize_vector(normal_vector):
    norm = torch.norm(normal_vector, dim=-1, keepdim=True)
    norm[norm == 0] = 1
    normal_vector = normal_vector / norm
    return normal_vector


def compute_normal_vectors_triangle(xyz, knn_idx):
    """
    Calculate the normal vectors of the point cloud using the concept of triangular facets.
    @param xyz: torch.Tensor, shape(2, 2048, 3)
    @param knn_idx: torch.Tensor, shape(2, 2048, 32)
    @return: normal voctor, torch.Tensor, shape(2, 2048, 3)
    """
    batch_size, point_num = xyz.shape[0], xyz.shape[1]
    normals = torch.zeros_like(xyz)
    for i in range(batch_size):
        single_xyz = xyz[i]
        single_knn_idx = knn_idx[i]
        for j in range(point_num):
            v1 = single_xyz[single_knn_idx[j, 0], :3]
            v2 = single_xyz[single_knn_idx[j, 1], :3]
            v3 = single_xyz[single_knn_idx[j, 2], :3]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = torch.cross(edge1, edge2)
            normals[i, single_knn_idx[j, 0]] += normal
            normals[i, single_knn_idx[j, 1]] += normal
            normals[i, single_knn_idx[j, 2]] += normal
        normals[i,] = normalize_vector(normals[i,])
    return normals


def Euclidean_Space_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def geodesic_distance(src):
    src_mean = torch.mean(src, dim=-1, keepdim=True)
    src = src - src_mean
    src = src / src.norm(p=2, dim=-1, keepdim=True)
    inner = torch.matmul(src, src.transpose(-2, -1))
    dist = torch.acos(inner)
    dist[torch.isnan(dist)] = 0
    return dist


def Euclidean_Space_distance_Dataset(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * np.matmul(src, dst.T)
    dist += np.sum(src ** 2, -1).reshape(N, 1)
    dist += np.sum(dst ** 2, -1).reshape(1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = Euclidean_Space_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(n_sample, radius, n_neighbor, point_xyz, point_feature, point_normal, knn=False):
    """
    sample point cloud and group feature
    @param n_sample: Number of points retained after downsampling
    @param radius: Sphere query radius
    @param n_neighbor: Number of neighboring points queried
    @param point_xyz: xyz information, [B, N, 3]
    @param point_feature: feature information, [B, N, C]
    @param point_normal: normal vector, [B, N, 3]
    @param knn: whether to use knn
    @return: new_xyz: (B, n_sample, 3); group feature: (B, n_sample, n_neighbor, C+3)
    """
    B, N, C = point_xyz.shape
    S = n_sample
    fps_id = farthest_point_sample(point_xyz, n_sample)
    new_xyz = index_points(point_xyz, fps_id)
    new_normal = index_points(point_normal, fps_id)
    if knn:
        dists = Euclidean_Space_distance(new_xyz, point_xyz)
        group_id = dists.argsort()[:, :, :n_neighbor]
    else:
        group_id = query_ball_point(radius, n_neighbor, point_xyz, new_xyz)
    grouped_xyz = index_points(point_xyz, group_id)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if point_feature is not None:
        grouped_feature = index_points(point_feature, group_id)
        grouped_feature = torch.cat([grouped_xyz_norm, grouped_feature], dim=-1)
    else:
        new_feature = grouped_xyz_norm
    return new_xyz, grouped_feature, new_normal


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_sample, radius, n_neighbor, in_channel, mlp, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.n_sample = n_sample
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.knn = knn
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.conv_list.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bn_list.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, point_xyz, point_feature, point_normal):
        new_xyz, grouped_feature, new_normal = sample_and_group(self.n_sample, self.radius, self.n_neighbor,
                                                                point_xyz, point_feature, point_normal,
                                                                self.knn)
        grouped_feature = grouped_feature.permute(0, 3, 2, 1)
        for conv, bn in list(zip(self.conv_list, self.bn_list)):
            grouped_feature = F.relu(bn(conv(grouped_feature)))
        new_feature = torch.max(grouped_feature, 2)[0].transpose(1, 2)
        return new_xyz, new_feature, new_normal


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.conv_list.append(nn.Conv1d(last_channel, out_channel, 1))
            self.bn_list.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, point_xyz_few, point_feature_few, point_xyz_large, point_feature_large):
        B, N, C = point_xyz_large.shape
        _, S, _ = point_xyz_few.shape
        if S == 1:
            interpolated_points = point_feature_few.repeat(1, N, 1)
        else:
            dists = Euclidean_Space_distance(point_xyz_large, point_xyz_few)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(point_feature_few, idx) * weight.view(B, N, 3, 1), dim=2)
        if point_feature_large is not None:
            new_point_feature_large = torch.cat([point_feature_large, interpolated_points], dim=-1)
        else:
            new_point_feature_large = interpolated_points
        new_point_feature_large = new_point_feature_large.permute(0, 2, 1)
        for conv, bn in list(zip(self.conv_list, self.bn_list)):
            new_point_feature_large = F.relu(bn(conv(new_point_feature_large)))
        return new_point_feature_large


class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)
