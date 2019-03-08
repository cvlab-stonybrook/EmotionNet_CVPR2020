# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# TODO: this is modified from main_mclass_corss_entropy_v2.py
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 11:09
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)
from sklearn.metrics.pairwise import cosine_similarity
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
import CNNs.models as models
import CNNs.utils.util as CNN_utils
from torch.optim import lr_scheduler
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





    print_func = print


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



    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # model = torch.nn.DataParallel(model).cuda()
            model = model.cuda()



    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            import collections
            if isinstance(checkpoint, collections.OrderedDict):
                load_state_dict(model, checkpoint, exclude_layers=['fc.weight', 'fc.bias'])


            else:
                load_state_dict(model, checkpoint['state_dict'], exclude_layers=['module.fc.weight', 'module.fc.bias'])
                print_func("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        print_func("=> This script is for fine-tuning only, please double check '{}'".format(args.resume))
        print_func("Now using randomly initialized parameters!")

    cudnn.benchmark = True




    from PyUtils.pickle_utils import loadpickle
    from PublicEmotionDatasets.Unbiasedemotion.constants import emotion2idx, idx2emotion
    from PyUtils.dict_utils import string_list2dict
    import numpy as np
    from torchvision.datasets.folder import default_loader
    tag_wordvectors = loadpickle(args.tag_embeddings)
    tag_words = []
    tag_matrix = []
    label_words = []
    label_matrix = []

    for x_tag in tag_wordvectors:
        tag_words.append(x_tag)
        tag_matrix.append(tag_wordvectors[x_tag])
        if x_tag in emotion2idx:
            label_words.append(x_tag)
            label_matrix.append(tag_wordvectors[x_tag])
    idx2tag, tag2idx= string_list2dict(tag_words)
    idx2label, label2idx = string_list2dict(label_words)
    tag_matrix = np.array(tag_matrix)
    label_matrix = np.array(label_matrix)
    label_matrix = label_matrix.squeeze(1)
    tag_matrix = tag_matrix.squeeze(1)
    val_list = loadpickle(args.val_file)
    from CNNs.datasets.multilabel import get_val_simple_transform
    val_transform = get_val_simple_transform()
    model.eval()

    correct = 0
    total = len(val_list) * 1.0
    for i, (input_image_file, target) in enumerate(val_list):
        # measure data loading time

        image_path = os.path.join(args.data_dir, input_image_file)
        input_image = default_loader(image_path)
        input_image = val_transform(input_image)


        if args.gpu is not None:
            input_image = input_image.cuda(args.gpu, non_blocking=True)
        input_image = input_image.unsqueeze(0).cuda()

        # target_idx = target.nonzero() [:,1]


        # compute output
        output, output_proj = model(input_image)

        output_proj = output_proj.cpu().data.numpy()

        dot_product_label = cosine_similarity(output_proj, label_matrix)[0]
        output_label = np.argmax(dot_product_label)
        if output_label == target:
            correct += 1

        dot_product_tag = cosine_similarity(output_proj, tag_matrix)[0]
        out_tags = np.argsort(dot_product_tag)[::-1][:10]

        print("* {} Image: {} GT label: {}, predicted label: {} \{}".format(i, input_image_file, idx2emotion[target], idx2label[output_label], "Correct" if output_label == target else "Wrong"))
        print(" == closest tags: {}".format(', '.join(['{}({:.02f})'.format(idx2tag[x], dot_product_tag[x]) for x in out_tags])))
    print("Accuracy {:.4f}".format(correct/total))








if __name__ == '__main__':
    main()

