# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): feature extractor, slow version, double check!
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 11:09
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)

# import argparse
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
from CNNs.utils.logger import Logger
import CNNs.losses as loss_funcs
from torch.optim import lr_scheduler
import logging
from CNNs.dataloaders.utils import none_collate
from PyUtils.file_utils import get_date_str, get_dir, get_stem
from PyUtils import log_utils
import CNNs.datasets as custom_datasets
from CNNs.utils.config import parse_config
from CNNs.models.resnet import load_state_dict
import torch.nn.functional as F
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_instance(module, name, args):
    return getattr(module, name)(args)


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Image CNN training from Configure Files")
    parser.add_argument('--config_file', required=True, help="This scripts only accepts parameters from Json files")
    input_args = parser.parse_args()

    config_file = input_args.config_file

    args = parse_config(config_file)
    if args.name is None:
        args.name = get_stem(config_file)

    torch.set_default_tensor_type('torch.FloatTensor')
    best_prec1 = 0

    args.script_name = get_stem(__file__)
    current_time_str = get_date_str()
    if args.save_directory is None:
        save_directory = get_dir(os.path.join(project_root, args.ckpts_dir, '{:s}'.format(args.name), '{:s}-{:s}'.format(args.ID, current_time_str)))
    else:
        save_directory = get_dir(os.path.join(project_root, args.ckpts_dir, args.save_directory))

    print("Save to {}".format(save_directory))
    log_file = os.path.join(save_directory, 'log-{0}.txt'.format(current_time_str))
    logger = log_utils.get_logger(log_file)
    log_utils.print_config(vars(args), logger)


    print_func = logger.info
    print_func('ConfigFile: {}'.format(config_file))
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
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    if args.pretrained:
        print_func("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=args.num_classes)
    else:
        print_func("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False, num_classes=args.num_classes)

    if args.freeze:
        model = CNN_utils.freeze_all_except_fc(model)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        print_func('Please only specify one GPU since we are working in batch size 1 model')
        return


    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            import collections
            if not args.evaluate:
                if isinstance(checkpoint, collections.OrderedDict):
                    load_state_dict(model, checkpoint, exclude_layers=['fc.weight', 'fc.bias'])


                else:
                    load_state_dict(model, checkpoint['state_dict'], exclude_layers=['module.fc.weight', 'module.fc.bias'])
                    print_func("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
            else:
                if isinstance(checkpoint, collections.OrderedDict):
                    load_state_dict(model, checkpoint,strict=True)


                else:
                    load_state_dict(model, checkpoint['state_dict'],strict=True)
                    print_func("=> loaded checkpoint '{}' (epoch {})"
                               .format(args.resume, checkpoint['epoch']))
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        print_func("=> This script is for fine-tuning only, please double check '{}'".format(args.resume))
        print_func("Now using randomly initialized parameters!")

    cudnn.benchmark = True

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_func("Total Parameters: {0}\t Gradient Parameters: {1}".format(model_total_params, model_grad_params))

    # Data loading code
    # val_dataset = get_instance(custom_datasets, '{0}'.format(args.valloader), args)
    from PyUtils.pickle_utils import loadpickle
    from torchvision.datasets.folder import default_loader

    val_dataset = loadpickle(args.val_file)
    image_directory = args.data_dir
    from CNNs.datasets.multilabel import get_val_simple_transform
    val_transform = get_val_simple_transform()
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



        image_path = os.path.join(image_directory, s_data[0])

        try:
            input_image = default_loader(image_path)
        except:
            print("WARN: {} Problematic!, Skip!".format(image_path))

            continue

        input_image = val_transform(input_image)


        if args.gpu is not None:
            input_image = input_image.cuda(args.gpu, non_blocking=True)

        output = model(input_image.unsqueeze_(0))
        output = output.cpu().data.numpy()
        # image_rel_path = os.path.join(*(s_image_name.split(os.sep)[-int(args.rel_path_depth):]))

        if args.individual_feat:
            if image_directory in created_paths:
                np.save(os.path.join(feature_save_directory, '{}.npy'.format(s_data[0])), output)
            else:
                get_dir(os.path.join(feature_save_directory, image_directory))
                np.save(os.path.join(feature_save_directory, '{}.npy'.format(s_data[0])), output)
                created_paths.add(image_directory)
        else:
            data_dict[s_data[0]] = output
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