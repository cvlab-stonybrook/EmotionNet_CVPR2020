# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 11:09
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import CNNs.models as models
from CNNs.utils.config import parse_config
from PyUtils.pickle_utils import loadpickle
from torchvision.datasets.folder import default_loader
from CNNs.dataloaders.transformations import get_val_simple_transform, get_train_simple_transform
import numpy as np
import tqdm
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_instance(module, name, args, **kwargs):
    return getattr(module, name)(args, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Image CNN Verification")
    parser.add_argument('--config_file', required=True, help="This scripts only accepts parameters from Json files")
    input_args = parser.parse_args()

    config_file = input_args.config_file

    args = parse_config(config_file)

    torch.set_default_tensor_type('torch.FloatTensor')

    model = models.__dict__[args.arch](pretrained=False, num_classes=args.num_classes)

    model = model.cuda(args.gpu)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove the module. in the state dict
            from collections import OrderedDict
            parallel_state_dict = checkpoint['state_dict']
            local_state_dict = OrderedDict()
            for k, v in parallel_state_dict.items():
                name = k[7:]  # remove `module.`
                local_state_dict[name] = v
            model.load_state_dict(local_state_dict, strict=True)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    model_total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters: {0}".format(model_total_params))

    image_transformation = get_val_simple_transform()
    val_dataset = loadpickle(args.val_data)
    import random
    random.shuffle(val_dataset)
    val_image_directory = args.val_image_directory
    idx2key = loadpickle(args.key_idx_correspondences)['idx2key']
    model.eval()
    for idx, s_data_item in enumerate(val_dataset):
        s_image_cid, s_image_rel_path, s_image_searchkeys, s_image_tags, s_image_url = s_data_item

        s_image_path = os.path.join(val_image_directory, s_image_rel_path)
        if not os.path.exists(s_image_path):
            print("{} Not Exist".format(s_image_rel_path))

        s_image = default_loader(s_image_path)
        input = image_transformation(s_image)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True).unsqueeze_(0)

        output = F.softmax(model(input), dim=1)
        # output = model(input)
        output = output.squeeze_(0).cpu().data.numpy()
        output_topK = np.argsort(output)[::-1][:10]

        output_string = ', '.join(['{}({:.04f})'.format(idx2key[x], float(output[x])) for x in output_topK])
        print("** {}\t{}\t{}".format(idx, s_image_cid, s_image_url))
        print("Search Keys: {}".format(', '.join(s_image_searchkeys)))
        print("Predicted Keys: {}".format(output_string))


if __name__ == '__main__':
    main()

