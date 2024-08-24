# -*- coding: utf-8 -*-
# @Time    : 2023-05-14
# @Author  : lab
# @desc    :
import datetime
import logging
from pathlib import Path

import torch


def create_logger(args, log_dir):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model.split("/")[1]))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_string(logger, str_print):
    logger.info(str_print)
    print(str_print)


def create_dir(args, name):
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model.split("/")[0]).joinpath(name)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(time_str)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    return log_dir, exp_dir, checkpoints_dir


def cls2onehot(y, num_classes, device_main):
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.to(device_main)
    return new_y


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
