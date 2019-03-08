# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 02/Nov/2018 14:27

import os
import torch
from CNNs.dataloaders.transformations import *
from PyUtils.pickle_utils import loadpickle
from CNNs.dataloaders.imagename_loader import ImageNamesRelLists


def feature_list(args, annotation_file, data_dir, rel_path_h=None):
    image_paths = loadpickle(annotation_file)

    dataset = ImageNamesRelLists(image_paths=image_paths, image_root=data_dir, transform=get_val_simple_transform())

    return dataset




