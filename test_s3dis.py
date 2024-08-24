import argparse
import importlib

import numpy as np
from tqdm import tqdm

from dataset.S3DIS.S3DISDataLoader import S3DISDatasetWholeScene
from utils.main_utils import *

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('s3dis testing')
    parser.add_argument('--scene_batch_size', type=int, default=1, help='scene batch size in testing [must: 1]')
    parser.add_argument('--point_batch_size', type=int, default=2, help='point batch size in testing [default: 2]')
    parser.add_argument('--gpu', type=str, default=[0], help='specify gpu device')
    parser.add_argument('--model', type=str, default='PointTransformer/PointTransformer')
    parser.add_argument('--data_path', default="data/S3DISNPY", type=str, help='path store data')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--input_dim', type=int, default=12, help='point dimension')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--class_num', type=int, default=13, help='category number')
    parser.add_argument('--num_neighbor', type=int, default=24, help='knn neighbor')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--process_data', action='store_true', default=True, help='use normals')
    parser.add_argument('--block_size', action='store_true', default=1.0, help='size for cropping from scene')
    parser.add_argument('--stride', action='store_true', default=0.5, help='length of step forward')
    parser.add_argument('--sample_rate', action='store_true', default=1.0, help='')
    parser.add_argument('--padding', action='store_true', default=0.001, help='edge overflow for cropping')
    parser.add_argument('--visual', action='store_true', default=True, help='Weather to visualize the prediction')
    parser.add_argument('--num_votes', type=int, default=5, help='aggregate scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """
    Some points might be predicted multiple times, so each prediction result for each point should be accumulated
    in the `vote_label_pool`. Finally, take the maximum value from `vote_label_pool` to get the final prediction
    result for each point.
    :param vote_label_pool: (1047554, 13)
    :param point_idx: (2, 2048)
    :param pred_label: (2, 2048)
    :param weight: (2, 2048)
    :return:
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not torch.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------------------------------------------------------
    device_main = torch.device('cuda:{}'.format(args.gpu[0]))
    # ------------------------------------------------------------------------------------------------------------------
    # Create a directory
    # ------------------------------------------------------------------------------------------------------------------
    log_dir, exp_dir, checkpoints_dir = create_dir(args, 'semseg/test')
    # ------------------------------------------------------------------------------------------------------------------
    # Create an output log
    # ------------------------------------------------------------------------------------------------------------------
    logger = create_logger(args, log_dir)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)
    log_string(logger, 'Load dataset ...')
    test_dataset = S3DISDatasetWholeScene(args=args, split="test")
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.scene_batch_size, shuffle=False,
                                                 num_workers=0, drop_last=False)
    # ------------------------------------------------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------------------------------------------------
    model = importlib.import_module('models.{}.{}'.format(args.model.split("/")[0], args.model.split("/")[1]))
    classifier = model.get_model(args).to(device_main)
    # ------------------------------------------------------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------------------------------------------------------
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger, 'Use pretrain model')
    except FileNotFoundError as e:
        log_string(logger, 'No existing model, starting training from scratch...')
    # ------------------------------------------------------------------------------------------------------------------
    # Start testing
    # ------------------------------------------------------------------------------------------------------------------
    with torch.no_grad():
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(args.class_num)]
        total_correct_class = [0 for _ in range(args.class_num)]
        total_iou_deno_class = [0 for _ in range(args.class_num)]
        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            whole_scene_label = test_dataset.scene_lable_list[i]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], args.class_num))
            for j in range(args.num_votes):
                scene_data = data[0].squeeze().to(device_main)
                scene_normal = data[1].squeeze().to(device_main)
                scene_weight = data[3].squeeze().to(device_main)
                point_index = data[4].squeeze().to(device_main)
                num_blocks = scene_data.shape[0]
                # ------------------------------------------------------------------------------------------------------
                # The entire point cloud scene is divided into `num_blocks` blocks. Each time, `point_batch_size` blocks
                # are taken, so the entire point cloud scene is fetched in `fetch_number` iterations.
                # ------------------------------------------------------------------------------------------------------
                fetch_number = int(np.ceil(num_blocks / args.point_batch_size))
                for start_fetch in range(fetch_number):
                    start_idx = start_fetch * args.point_batch_size
                    end_idx = min((start_fetch + 1) * args.point_batch_size, num_blocks)
                    batch_data = scene_data[start_idx:end_idx, ...]
                    batch_normal = scene_normal[start_idx:end_idx, ...]
                    batch_weight = scene_weight[start_idx:end_idx, ...]
                    batch_point_index = point_index[start_idx:end_idx, ...]
                    # --------------------------------------------------------------------------------------------------
                    # Start prediction.
                    # --------------------------------------------------------------------------------------------------
                    batch_data = batch_data.float()
                    batch_normal = batch_normal.float()
                    input_data = torch.cat((batch_data, batch_normal), dim=-1)
                    seg_pred = classifier(input_data)
                    batch_seg_pred = seg_pred.data.max(-1)[1]
                    # --------------------------------------------------------------------------------------------------
                    # vote_label_pool: Count how many times each point in the scene is predicted in each category.
                    # batch_point_index: Indicate the index of each point in the scene.
                    # batch_seg_pred: Indicate the predicted label for each point.
                    # batch_weight: Indicate the weight associated with each point.
                    # --------------------------------------------------------------------------------------------------
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index, batch_seg_pred, batch_weight)
            # ----------------------------------------------------------------------------------------------------------
            # In the current category, the point with the highest number of predictions is most likely to belong to
            # that category.
            # ----------------------------------------------------------------------------------------------------------
            pred_choice = np.argmax(vote_label_pool, 1)
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the accuracy for all point clouds: `instance_acc`.
            # ----------------------------------------------------------------------------------------------------------
            correct = np.sum((pred_choice == whole_scene_label))
            total_correct += correct
            total_seen += whole_scene_label.shape[0]
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the average IoU.
            # ----------------------------------------------------------------------------------------------------------
            for l in range(args.class_num):
                total_seen_class[l] += np.sum((whole_scene_label == i))
                total_correct_class[l] += np.sum((pred_choice == l) & (whole_scene_label == l))
                total_iou_deno_class[l] += np.sum(((pred_choice == l) | (whole_scene_label == l)))
        IoU_each_class = np.array(total_correct_class) / (np.array(total_iou_deno_class) + 1e-6)
        # --------------------------------------------------------------------------------------------------------------
        # mIoU
        # --------------------------------------------------------------------------------------------------------------
        mIoU = np.mean(IoU_each_class)
        # --------------------------------------------------------------------------------------------------------------
        # OA
        # --------------------------------------------------------------------------------------------------------------
        instance_acc = total_correct / float(total_seen)
        # --------------------------------------------------------------------------------------------------------------
        # mAcc
        # --------------------------------------------------------------------------------------------------------------
        class_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class) + 1e-6))
        for i in range(args.class_num):
            log_string(logger, 'eval mIoU of %s %f' % (seg_label_to_cat[i] + ' ' * (14 - len(seg_label_to_cat[i])),
                                                       IoU_each_class[i]))
        log_string(logger, 'Testing mIoU: {}, OA: {}, mAcc: {}'.format(mIoU,
                                                                       instance_acc,
                                                                       class_acc))
        return mIoU, instance_acc, class_acc


if __name__ == '__main__':
    args = parse_args()
    main(args)
