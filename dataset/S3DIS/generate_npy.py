import glob
import os

import numpy as np
import open3d as o3d


def get_pcd(points):
    """
    Data type conversion，numpy -> point cloud
    @param points:
    @return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def compute_normal_vectors_open3d(xyz, neighbor_num):
    """
    Calculate the normal vectors of the point cloud by using the built-in functions in Open3D.
    @param xyz: torch, shape(2048, 3)
    @return: 法向量, shape(2048, 3)
    """
    single_pcd = get_pcd(xyz)
    single_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=neighbor_num))
    single_pcd_normal = np.asarray(single_pcd.normals)
    return single_pcd_normal


def collect_point_label(anno_path, out_filename, s3dis_classes, s3dis_class2label):
    """
    Assign labels
    :param anno_path:
    :param out_filename:
    :param s3dis_classes:
    :param s3dis_class2label:
    :return:
    """
    points_list = []
    for single_point_file in glob.glob(os.path.join(anno_path, '*.txt')):
        class_name = os.path.basename(single_point_file).split('_')[0]
        if class_name not in s3dis_classes:
            class_name = 'clutter'
        point_data = np.loadtxt(single_point_file)
        point_normal = compute_normal_vectors_open3d(point_data[:, :3], 20)
        point_label = np.ones((point_data.shape[0], 1)) * s3dis_class2label[class_name]
        points_list.append(np.concatenate([point_data, point_normal, point_label], 1))
    point_data_with_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(point_data_with_label, axis=0)[0:3]
    point_data_with_label[:, 0:3] -= xyz_min
    np.save(out_filename, point_data_with_label)


def main(input_folder, output_folder):
    s3dis_classes = [x.rstrip() for x in open("meta/class_names.txt")]
    s3dis_class2label = {cls: i for i, cls in enumerate(s3dis_classes)}
    anno_paths = [line.rstrip() for line in open("meta/anno_paths.txt")]
    anno_paths = [os.path.join(input_folder, p) for p in anno_paths]
    # ------------------------------------------------------------------------------------------------------------------
    # Iterate through and save the txt file as an npy file.
    # ------------------------------------------------------------------------------------------------------------------
    for anno_path in anno_paths:
        if os.path.isdir(anno_path):
            elements = anno_path.split('/')
            out_file_name = elements[-3] + '_' + elements[-2] + '.npy'
            out_file_path = os.path.join(output_folder, out_file_name)
            collect_point_label(anno_path, out_file_path, s3dis_classes, s3dis_class2label)


if __name__ == '__main__':
    data_path = r"../../data/S3DISTXT/"
    save_path = "../../data/S3DISNPY/"
    main(data_path, save_path)
