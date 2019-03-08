# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): feature extractor, slow version, double check!
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 11:09
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)

import argparse
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import CNNs.dataloaders as datasets
import CNNs.models as models
import CNNs.utils.util as CNN_utils
import CNNs.options.basic_parser as parser
from CNNs.utils.logger import Logger
import CNNs.losses as loss_funcs
from torch.optim import lr_scheduler
import logging
from CNNs.dataloaders.utils import none_collate
from PyUtils.file_utils import get_date_str, get_dir, get_stem
from PyUtils import log_utils
import CNNs.datasets as custom_datasets
from CNNs.utils.config import parse_config

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_instance(module, name, args, **kwargs):
    return getattr(module, name)(args, **kwargs)


def main():
    best_prec1 = 0

    args = parser.parser.parse_args()
    if args.config is not None:
        args = parse_config(args.config)

    script_name_stem = get_stem(__file__)
    current_time_str = get_date_str()

    if args.save_directory is None:
        raise FileNotFoundError("Saving directory should be specified for feature extraction tasks")
    save_directory = get_dir(args.save_directory)

    print("Save to {}".format(save_directory))
    log_file = os.path.join(save_directory, 'log-{0}.txt'.format(current_time_str))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)
    print_func = logger.info
    args.log_file = log_file

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)


    if args.arch == 'resnet50_feature_extractor':
        print_func("=> using pre-trained model '{}' to LOAD FEATURES".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=args.num_classes, param_name=args.paramname)

    else:
        print_func("This is only for feature extractors!, Please double check the parameters!")
        return

    # if args.freeze:
    #     model = CNN_utils.freeze_all_except_fc(model)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        print_func('Please only specify one GPU since we are working in batch size 1 model')
        return

    cudnn.benchmark = True

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_func("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))

    # Data loading code
    val_dataset = get_instance(custom_datasets, '{0}_val'.format(args.dataset.name), args, **args.dataset.args)
    import tqdm
    import numpy as np

    if args.individual_feat:
        feature_save_directory = get_dir(os.path.join(save_directory, 'individual-features'))
        created_paths = set()
    else:
        data_dict = {}
        feature_save_directory = os.path.join(save_directory, 'feature.pkl')

    model.eval()
    for s_data in tqdm.tqdm(val_dataset, desc="Extracting Features"):
        if s_data is None:
            continue
        s_image_name = s_data[1]
        s_image_data = s_data[0]
        if args.gpu is not None:
            s_image_data = s_image_data.cuda(args.gpu, non_blocking=True)

        output = model(s_image_data.unsqueeze_(0))
        output = output.cpu().data.numpy()
        image_rel_path = os.path.join(*(s_image_name.split(os.sep)[-args.rel_path_depth:]))

        if args.individual_feat:
            image_directory = os.path.dirname(image_rel_path)
            if image_directory in created_paths:
                np.save(os.path.join(feature_save_directory, '{}.npy'.format(image_rel_path)), output)
            else:
                get_dir(os.path.join(feature_save_directory, image_directory))
                np.save(os.path.join(feature_save_directory, '{}.npy'.format(image_rel_path)), output)
                created_paths.add(image_directory)
        else:
            data_dict[image_rel_path] = output
        # image_name = os.path.basename(s_image_name)
        #
        # if args.individual_feat:
        #         # image_name = os.path.basename(s_image_name)
        #
        #         np.save(os.path.join(feature_save_directory, '{}.npy'.format(image_name)), output)
        #         # created_paths.add(image_directory)
        # else:
        #         data_dict[get_stem(image_name)] = output

    if args.individual_feat:
        print_func("Done")
    else:
        from PyUtils.pickle_utils import save2pickle
        print_func("Saving to a single big file!")

        save2pickle(feature_save_directory, data_dict)
        print_func("Done")












if __name__ == '__main__':
    main()