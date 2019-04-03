# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 15:07
from PyUtils.pickle_utils import loadpickle
from CNNs.dataloaders.sample_loader import SampleLoader
from CNNs.dataloaders.transformations import *


def categorical_train(args):
    image_information = loadpickle(args.train_file)
    category_counts = {}
    for s_idx in image_information:
        category_counts[s_idx] = len(image_information[s_idx])
    dataset = SampleLoader(categories=image_information, categories_counts=category_counts, root=args.data_dir, transform=get_train_simple_transform(), target_transform=None, sample_size=args.sample_size)
    return dataset