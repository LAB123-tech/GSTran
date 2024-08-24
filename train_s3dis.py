import argparse
import importlib

import numpy as np
from tqdm import tqdm

from dataset.S3DIS.S3DISDataLoader import S3DISDataset, my_collate_fn
from utils.main_utils import *

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
seg_classes = {cls: i for i, cls in enumerate(classes)}
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('s3dis training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--gpu', type=str, default=[0], help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', type=str, default='PointTransformer/PointTransformer')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--data_path', default="data/S3DISNPY", type=str, help='path store data')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--input_dim', type=int, default=12, help='point dimension')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--class_num', type=int, default=13, help='Classification Number')
    parser.add_argument('--num_neighbor', type=int, default=24, help='knn neighbor')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--process_data', action='store_true', default=True, help='use normals')
    parser.add_argument('--block_size', action='store_true', default=1.0, help='use normals')
    return parser.parse_args()


def train(classifier, trainDataLoader, optimizer, scheduler, criterion, logger, args, device_main, epoch):
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    classifier = classifier.train()
    for i, (points, normal, seg_label, weights) in tqdm(enumerate(trainDataLoader),
                                                        total=len(trainDataLoader),
                                                        smoothing=0.9):
        optimizer.zero_grad()
        points = points.float().to(device_main)
        seg_label = seg_label.long().to(device_main)
        normal = normal.long().to(device_main)
        weights = weights.to(device_main)
        # --------------------------------------------------------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------------------------------------------------------
        seg_pred = classifier(torch.cat((points, normal), dim=-1))
        loss = criterion(seg_pred.contiguous().view(-1, args.class_num), seg_label.view(-1, 1)[:, 0],
                         weights.view(-1, 1)[:, 0])
        loss.backward()
        optimizer.step()
        pred_choice = seg_pred.view(-1, args.class_num).max(1)[1]
        batch_label = seg_label.view(-1, 1)[:, 0]
        correct = (pred_choice == batch_label).sum().item()
        total_correct += correct
        total_seen += (args.batch_size * args.num_point)
        loss_sum += loss
    scheduler.step(epoch)
    train_instance_acc = total_correct / float(total_seen)
    loss_mean = loss_sum / len(trainDataLoader)
    log_string(logger, 'Epoch: {}, Loss: {}, Training Instance Accuracy: {}'.format(epoch,
                                                                                    loss_mean,
                                                                                    train_instance_acc))
    return train_instance_acc


def test(classifier, testDataLoader, logger, criterion, args, device_main, epoch):
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(args.class_num)]
    total_correct_class = [0 for _ in range(args.class_num)]
    total_iou_deno_class = [0 for _ in range(args.class_num)]
    classifier = classifier.eval()
    with torch.no_grad():
        for i, (points, normal, seg_label, weights) in tqdm(enumerate(testDataLoader),
                                                            total=len(testDataLoader),
                                                            smoothing=0.9):
            points = points.float().to(device_main)
            seg_label = seg_label.long().to(device_main)
            normal = normal.long().to(device_main)
            weights = weights.to(device_main)
            # --------------------------------------------------------------------------------------------------------------
            # 预测
            # --------------------------------------------------------------------------------------------------------------
            seg_pred = classifier(torch.cat((points, normal), dim=-1))
            loss = criterion(seg_pred.contiguous().view(-1, args.class_num), seg_label.view(-1, 1)[:, 0],
                             weights.view(-1, 1)[:, 0])
            loss_sum += loss
            pred_choice = seg_pred.view(-1, args.class_num).max(1)[1]
            batch_label = seg_label.view(-1, 1)[:, 0]
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the accuracy for all point clouds: instance_acc
            # ----------------------------------------------------------------------------------------------------------
            correct = (pred_choice == batch_label).sum().item()
            total_correct += correct
            total_seen += (args.batch_size * args.num_point)
            # ----------------------------------------------------------------------------------------------------------
            # Calculate the average IoU
            # ----------------------------------------------------------------------------------------------------------
            for i in range(args.class_num):
                total_seen_class[i] += (batch_label == i).sum().item()
                total_correct_class[i] += ((pred_choice == i) & (batch_label == i)).sum().item()
                total_iou_deno_class[i] += ((pred_choice == i) | (batch_label == i)).sum().item()
        loss_mean = loss_sum / len(testDataLoader)
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
        log_string(logger, 'Epoch: {}, Loss: {}, Testing mIoU: {}, OA: {}, mAcc: {}'.format(epoch,
                                                                                            loss_mean,
                                                                                            mIoU,
                                                                                            instance_acc,
                                                                                            class_acc))
        return mIoU, instance_acc, class_acc


def main(args):
    # ------------------------------------------------------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------------------------------------------------------
    device_main = torch.device('cuda:{}'.format(args.gpu[0]))
    # ------------------------------------------------------------------------------------------------------------------
    # Create a directory
    # ------------------------------------------------------------------------------------------------------------------
    log_dir, exp_dir, checkpoints_dir = create_dir(args, 'semseg/train')
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
    train_dataset = S3DISDataset(args=args, split='train')
    test_dataset = S3DISDataset(args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, drop_last=False,
                                                  collate_fn=lambda x: my_collate_fn(x))
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, drop_last=False,
                                                 collate_fn=lambda x: my_collate_fn(x))
    # ------------------------------------------------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------------------------------------------------
    model = importlib.import_module('models.{}.{}'.format(args.model.split("/")[0], args.model.split("/")[1]))
    classifier = model.get_model(args).to(device_main)
    criterion = model.get_loss().to(device_main)
    # ------------------------------------------------------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------------------------------------------------------
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger, 'Use pretrain model')
    except FileNotFoundError as e:
        log_string(logger, 'No existing model, starting training from scratch...')
        start_epoch = 0
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
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0.00005)
    # ------------------------------------------------------------------------------------------------------------------
    # Set momentum
    # ------------------------------------------------------------------------------------------------------------------
    momentum_original = 0.1
    momentum_decay = 0.5
    momentum_decay_step = args.step_size
    best_instance_acc = 0
    best_class_acc = 0
    best_iou = 0
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        # --------------------------------------------------------------------------------------------------------------
        # Adjust momentum
        # --------------------------------------------------------------------------------------------------------------
        momentum = momentum_original * (momentum_decay ** (epoch // momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        train_instance_acc = train(classifier, trainDataLoader, optimizer, scheduler, criterion, logger,
                                   args, device_main, epoch)
        test_mIoU, test_instance_acc, test_class_acc = test(classifier, testDataLoader, logger, criterion, args,
                                                            device_main, epoch)
        # --------------------------------------------------------------------------------------------------------------
        # Update accuracy
        # --------------------------------------------------------------------------------------------------------------
        if test_mIoU > best_iou:
            best_iou = test_mIoU
        if test_instance_acc > best_instance_acc:
            best_instance_acc = test_instance_acc
        if test_class_acc > best_class_acc:
            best_class_acc = test_class_acc
        log_string(logger, 'Best eval mean IoU: %f' % best_iou)
        log_string(logger, 'Best eval instance accuracy: %f' % best_instance_acc)
        log_string(logger, 'Best eval class accuracy: %f' % best_class_acc)
        # --------------------------------------------------------------------------------------------------------------
        # Save model
        # --------------------------------------------------------------------------------------------------------------
        if test_mIoU >= best_iou:
            log_string(logger, 'Save best model...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string(logger, 'Saving at %s' % save_path)
            state = {'epoch': epoch,
                     'train_instance_acc': train_instance_acc,
                     'test_instance_acc': best_instance_acc,
                     'test_class_acc': best_class_acc,
                     'test_mIou': best_iou,
                     'model_state_dict': classifier.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, save_path)
        else:
            log_string(logger, 'Save last model...')
            save_path = str(checkpoints_dir) + '/last_model.pth'
            log_string(logger, 'Saving at %s' % save_path)
            state = {'epoch': epoch,
                     'train_instance_acc': train_instance_acc,
                     'test_instance_acc': test_instance_acc,
                     'test_class_acc': test_class_acc,
                     'test_mIou': test_mIoU,
                     'model_state_dict': classifier.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, save_path)
        log_string(logger, "------------------------------------------------------------------------------------------")
    log_string(logger, 'End of training...')


if __name__ == '__main__':
    main(parse_args())
