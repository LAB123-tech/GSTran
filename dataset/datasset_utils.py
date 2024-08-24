# -*- coding: utf-8 -*-
# @Time    : 2023-05-16
# @Author  : lab
# @desc    :
import numpy as np


def farthest_point_sample(point, n_point):
    """
    Input:
        xyz: point cloud data, [N, D]
        n_point: number of samples
    Return:
        centroids: sampled point cloud index, [n_point, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((n_point,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_point):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    index = centroids.astype(np.int32)
    return index


def svd_decomposition(point_cloud):
    """
    Calculate the covariance matrix of the point cloud, and use SVD (Singular Value Decomposition) to decompose the
     covariance matrix to obtain the eigenvalues and eigenvectors.
    :param point_cloud: N x 3
    :return eigenvalues, eigenvectors
    """
    centered_point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    covariance_matrix = np.dot(centered_point_cloud.T, centered_point_cloud) / (point_cloud.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    main_vector = eigenvectors[:, 0]
    return main_vector


def angle_between_vectors(a, b):
    """
    Calculate the angle between two vectors.
    :param a:
    :param b:
    :return:
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return radians


def correct_normal_direction(point_cloud):
    """
    Orient normal vectors
    @param point_cloud: ndarray, (2048, 6)
    @return:
    """
    point_cloud_data = point_cloud[:, :3]
    point_cloud_normal = point_cloud[:, 3:6]
    new_point_cloud_normal = np.zeros_like(point_cloud_normal)
    # ------------------------------------------------------------------------------------------------------------------
    # First, compute the initial normal vectors for the entire point cloud.
    # ------------------------------------------------------------------------------------------------------------------
    main_normal_vector = svd_decomposition(point_cloud_data)
    # ------------------------------------------------------------------------------------------------------------------
    # check if the normal vector of each point aligns with the direction of the principal normal vector.
    # If not, adjust it to align.
    # ------------------------------------------------------------------------------------------------------------------
    for i in range(point_cloud_data.shape[0]):
        single_normal_vector = point_cloud_normal[i]
        is_same_direction = (single_normal_vector * main_normal_vector).sum()
        if is_same_direction < 0:
            new_single_normal_vector = -single_normal_vector
        else:
            new_single_normal_vector = single_normal_vector
        new_point_cloud_normal[i] = new_single_normal_vector
    # ------------------------------------------------------------------------------------------------------------------
    # Then, calculate the deviation between each point's normal vector and the principal normal vector.
    # ------------------------------------------------------------------------------------------------------------------
    differences = np.zeros(new_point_cloud_normal.shape[0])
    for i in range(new_point_cloud_normal.shape[0]):
        differences[i] = angle_between_vectors(main_normal_vector, new_point_cloud_normal[i])
    return point_cloud_data, new_point_cloud_normal, differences
