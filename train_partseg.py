# -*- coding: utf-8 -*-
import argparse
import importlib
import os
import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.Accuracy import *
from dataset.ShapeNetPart.ShapeNetPartDataLoader import ShapeNetPartDataset, my_collate_fn
from utils.main_utils import *

# ----------------------------------------------------------------------------------------------------------------------
# {0:Airplane, 1:Airplane, ...49:Table}
# ----------------------------------------------------------------------------------------------------------------------
seg_classes = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15],
               'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23], 'Lamp': [24, 25, 26, 27],
               'Laptop': [28, 29], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
               'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('shapenet training')
    parser.add_argument('--batch_size', type=int, default=2, help='batch Size during training')
    parser.add_argument('--epoch', default=200, type=int, help='epoch to run')
    parser.add_argument('--gpu', type=str, default=[0], help='specify GPU devices')
    parser.add_argument('--model', type=str, default='PointTransformer/PointTransformer', help='model name')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--data_path', default="data/ShapeNetPart", type=str, help='path store data')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--input_dim', type=int, default=22, help='point Number')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--class_num', type=int, default=50, help='Classification Number')
    parser.add_argument('--num_category', type=int, default=16, help='category number')
    parser.add_argument('--num_neighbor', type=int, default=24, help='knn neighbor')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--use_uniform_sample', default=True, help='use uniform sampling')
    return parser.parse_args()


def train(classifier, trainDataLoader, optimizer, criterion, scheduler, logger, epoch, args, device_main, writer):
    train_metrics = {}
    mean_loss = []
    mean_acc = []
    mean_ins_iou = []
    mean_cls_iou = {single_class: [] for single_class in seg_classes.keys()}
    classifier = classifier.train()
    for batch_id, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()
        point_cloud_data = data[0].float().to(device_main)
        point_cloud_seg = data[1].long().to(device_main)
        point_cloud_cls = data[2].long().to(device_main)
        point_cloud_normal = data[3].float().to(device_main)
        # --------------------------------------------------------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------------------------------------------------------
        point_cloud_cls = cls2onehot(point_cloud_cls, args.num_category, device_main).repeat(1, args.num_point, 1)
        classifier_input = torch.cat([point_cloud_data, point_cloud_cls, point_cloud_normal], dim=-1)
        seg_pred = classifier(classifier_input)
        seg_true = point_cloud_seg
        loss = criterion(seg_pred.contiguous().view(-1, args.class_num), seg_true.view(-1, ))
        loss.backward()
        mean_loss.append(loss.item())
        optimizer.step()
        # --------------------------------------------------------------------------------------------------------------
        # Compute Acc, class_iou, instance_iou
        # --------------------------------------------------------------------------------------------------------------
        pred_choice = cal_pred_choice(seg_pred, seg_true)
        mean_acc = cal_acc(pred_choice, seg_true, mean_acc)
        mean_ins_iou, mean_cls_iou = cal_ins_cls_iou(pred_choice, seg_true, mean_ins_iou, mean_cls_iou)
    scheduler.step(epoch)
    train_loss = np.mean(mean_loss)
    train_acc = np.mean(mean_acc)
    train_ins_iou = np.mean(mean_ins_iou)
    # ------------------------------------------------------------------------------------------------------------------
    # First, calculate the average IoU for each class, and then compute the mean IoU across all classes.
    # ------------------------------------------------------------------------------------------------------------------
    for class_name in mean_cls_iou.keys():
        mean_cls_iou[class_name] = np.mean(mean_cls_iou[class_name])
    train_cls_iou = np.mean(list(mean_cls_iou.values()))
    train_metrics['instance_accuracy'] = train_acc
    train_metrics['class_iou'] = train_cls_iou
    train_metrics['instance_iou'] = train_ins_iou
    log_string(logger, 'Epoch: {}, Loss: {}, Train Instance Accuracy: {}, '
                       'Train Instance IoU: {}, Train Class IoU: {},'.format(epoch, train_loss,
                                                                             train_metrics['instance_accuracy'],
                                                                             train_metrics['class_iou'],
                                                                             train_metrics['instance_iou']))
    for class_name in sorted(mean_cls_iou.keys()):
        log_string(logger,
                   'eval mIoU of %s %f' % (class_name + ' ' * (14 - len(class_name)), mean_cls_iou[class_name]))
    # ------------------------------------------------------------------------------------------------------------------
    # Loss curve
    # ------------------------------------------------------------------------------------------------------------------
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_instance_accuracy", train_metrics['instance_accuracy'], epoch)
    writer.add_scalar("train_instance_iou", train_metrics['instance_iou'], epoch)
    writer.add_scalar("train_class_iou", train_metrics['class_iou'], epoch)
    return train_metrics


def test(classifier, testDataLoader, criterion, logger, epoch, args, device_main, writer):
    test_metrics = {}
    mean_loss = []
    mean_acc = []
    mean_ins_iou = []
    mean_cls_iou = {single_class: [] for single_class in seg_classes.keys()}
    classifier = classifier.eval()
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            point_cloud_data = data[0].float().to(device_main)
            point_cloud_seg = data[1].long().to(device_main)
            point_cloud_cls = data[2].long().to(device_main)
            point_cloud_normal = data[3].float().to(device_main)
            # --------------------------------------------------------------------------------------------------------------
            # Prediction
            # --------------------------------------------------------------------------------------------------------------
            point_cloud_cls = cls2onehot(point_cloud_cls, args.num_category, device_main).repeat(1, args.num_point, 1)
            classifier_input = torch.cat([point_cloud_data, point_cloud_cls, point_cloud_normal], dim=-1)
            seg_pred = classifier(classifier_input)
            seg_true = point_cloud_seg
            loss = criterion(seg_pred.contiguous().view(-1, args.class_num), point_cloud_seg.view(-1, ))
            mean_loss.append(loss.item())
            # --------------------------------------------------------------------------------------------------------------
            # Compute Acc, class_iou, instance_iou
            # --------------------------------------------------------------------------------------------------------------
            pred_choice = cal_pred_choice(seg_pred, seg_true)
            mean_acc = cal_acc(pred_choice, seg_true, mean_acc)
            mean_ins_iou, mean_cls_iou = cal_ins_cls_iou(pred_choice, seg_true, mean_ins_iou, mean_cls_iou)
    test_loss = np.mean(mean_loss)
    test_acc = np.mean(mean_acc)
    test_ins_iou = np.mean(mean_ins_iou)
    # ------------------------------------------------------------------------------------------------------------------
    # First, calculate the average IoU for each class, and then compute the mean IoU across all classes.
    # ------------------------------------------------------------------------------------------------------------------
    for class_name in mean_cls_iou.keys():
        mean_cls_iou[class_name] = np.mean(mean_cls_iou[class_name])
    test_cls_iou = np.mean(list(mean_cls_iou.values()))
    test_metrics['instance_accuracy'] = test_acc
    test_metrics['class_iou'] = test_cls_iou
    test_metrics['instance_iou'] = test_ins_iou
    log_string(logger, 'Epoch: {}, Loss: {}, Test Instance Accuracy: {}, '
                       'Test Instance IoU: {}, Test Class IoU: {},'.format(epoch, test_loss,
                                                                           test_metrics['instance_accuracy'],
                                                                           test_metrics['class_iou'],
                                                                           test_metrics['instance_iou']))
    for class_name in sorted(mean_cls_iou.keys()):
        log_string(logger,
                   'eval mIoU of %s %f' % (class_name + ' ' * (14 - len(class_name)), mean_cls_iou[class_name]))
    # ------------------------------------------------------------------------------------------------------------------
    # loss curve
    # ------------------------------------------------------------------------------------------------------------------
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_instance_accuracy", test_metrics['instance_accuracy'], epoch)
    writer.add_scalar("test_instance_iou", test_metrics['instance_iou'], epoch)
    writer.add_scalar("test_class_iou", test_metrics['class_iou'], epoch)
    return test_metrics


def main(args):
    # ------------------------------------------------------------------------------------------------------------------
    # tensorboard --logdir=runs/val
    # ------------------------------------------------------------------------------------------------------------------
    train_writer = SummaryWriter(os.path.join('runs', 'train'))
    test_writer = SummaryWriter(os.path.join('runs', 'val'))
    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------------------------------------------------------
    device_main = torch.device('cuda:{}'.format(args.gpu[0]))
    # ------------------------------------------------------------------------------------------------------------------
    # Create a directory
    # ------------------------------------------------------------------------------------------------------------------
    log_dir, exp_dir, checkpoints_dir = create_dir(args, 'partseg/train')
    # ------------------------------------------------------------------------------------------------------------------
    # Create an output log
    # ------------------------------------------------------------------------------------------------------------------
    logger = create_logger(args, log_dir)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)
    # ------------------------------------------------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------------------------------------------------
    log_string(logger, 'Load dataset ...')
    train_dataset = ShapeNetPartDataset(args=args, split='train_val')
    test_dataset = ShapeNetPartDataset(args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, drop_last=True, pin_memory=True,
                                                  collate_fn=lambda x: my_collate_fn(x, is_train=True))
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, drop_last=True, pin_memory=True,
                                                 collate_fn=lambda x: my_collate_fn(x, is_train=False))
    # ------------------------------------------------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------------------------------------------------
    model = importlib.import_module('models.{}.{}'.format(args.model.split("/")[0], args.model.split("/")[1]))
    classifier = model.get_model(args).to(device_main)
    criterion = model.get_loss().to(device_main)
    # ------------------------------------------------------------------------------------------------------------------
    # Optimizer + Learning Rate
    # ------------------------------------------------------------------------------------------------------------------
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0.00005)
    # ------------------------------------------------------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------------------------------------------------------
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth', map_location=device_main)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string(logger, 'Use pretrain model...')
    except FileNotFoundError:
        log_string(logger, 'No existing model, starting training from scratch...')
        start_epoch = 0
    # ------------------------------------------------------------------------------------------------------------------
    # Set momentum
    # ------------------------------------------------------------------------------------------------------------------
    momentum_original = 0.1
    momentum_decay = 0.5
    momentum_decay_step = args.step_size
    best_instance_acc = 0
    best_instance_iou = 0
    best_class_iou = 0
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        # --------------------------------------------------------------------------------------------------------------
        # Adjust momentum
        # --------------------------------------------------------------------------------------------------------------
        momentum = momentum_original * (momentum_decay ** (epoch // momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        train_metrics = train(classifier, trainDataLoader, optimizer, criterion, scheduler, logger, epoch, args,
                              device_main, train_writer)
        test_metrics = test(classifier, testDataLoader, criterion, logger, epoch, args, device_main, test_writer)
        # --------------------------------------------------------------------------------------------------------------
        # Update accuracy
        # --------------------------------------------------------------------------------------------------------------
        if test_metrics['instance_accuracy'] > best_instance_acc:
            best_instance_acc = test_metrics['instance_accuracy']
        if test_metrics['class_iou'] > best_class_iou:
            best_class_iou = test_metrics['class_iou']
        if test_metrics['instance_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['instance_iou']
        log_string(logger, 'Best Instance Accuracy is: %s' % str(best_instance_acc))
        log_string(logger, 'Best instance avg mIOU is: %s' % str(best_instance_iou))
        log_string(logger, 'Best class avg mIOU is: %s' % str(best_class_iou))
        # --------------------------------------------------------------------------------------------------------------
        # Save model
        # --------------------------------------------------------------------------------------------------------------
        if test_metrics['instance_iou'] >= best_instance_iou:
            log_string(logger, 'Save best model...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string(logger, 'Saving at %s' % save_path)
            state = {'epoch': epoch,
                     'train_acc': train_metrics['instance_accuracy'],
                     'test_acc': test_metrics['instance_accuracy'],
                     'class_avg_iou': test_metrics['class_iou'],
                     'instance_avg_iou': test_metrics['instance_iou'],
                     'model_state_dict': classifier.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, save_path)
        else:
            log_string(logger, 'Save last model...')
            save_path = str(checkpoints_dir) + '/last_model.pth'
            log_string(logger, 'Saving at %s' % save_path)
            state = {'epoch': epoch,
                     'train_acc': train_metrics['instance_accuracy'],
                     'test_acc': best_instance_acc,
                     'instance_avg_iou': best_instance_iou,
                     'model_state_dict': classifier.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, save_path)
        log_string(logger, "------------------------------------------------------------------------------------------")
    log_string(logger, 'End of training...')


if __name__ == '__main__':
    main(parse_args())
