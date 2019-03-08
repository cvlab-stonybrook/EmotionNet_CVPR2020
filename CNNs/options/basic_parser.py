# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 16:56


import argparse
import CNNs.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50init',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_patience', default=5, type=int,
                    help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
parser.add_argument('--lr_schedule', default=None, help='Manual LR scheduling e.g. 30,60,90')
parser.add_argument('--lr_min', default=0.00001, type=float, help='Minimum learning rate')


parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#TODO: don't forget this one!
parser.add_argument('--pretrained', dest='pretrained', default=True,
                    help='use pre-trained model')
parser.add_argument('--num_classes', default=1000, type=int, help='output classes')
parser.add_argument('--freeze', dest='freeze', action='store_true', help="Freezing early layers as feature extractors")

parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. assume only 1')

# TODO: my own stuff:
parser.add_argument('--ID', default='Unspecified', type=str, help="specify the name for experiment")
parser.add_argument('--config', default=None, type=str, help='the configuration files')
# check https://discuss.pytorch.org/t/multigpu-forward-pass/8865/4
# TODO: test this!
parser.add_argument('--device', default=None, type=str,
                    help='select on specific device on system level, different than --gpu, use before import torch, format: 0, 1, 2')