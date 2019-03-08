# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 17:25


# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 17:17

import torch.utils.data as data

from PIL import Image

import os
import os.path
# This is directly calling Pytorch's version
from torchvision.datasets.folder import accimage_loader, default_loader

# def has_file_allowed_extension(filename, extensions):
#     """Checks if a file is an allowed extension.
#
#     Args:
#         filename (string): path to a file
#
#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     filename_lower = filename.lower()
#     return any(filename_lower.endswith(ext) for ext in extensions)
#
#
# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx
#
#
# def make_dataset(dir, class_to_idx, extensions):
#     images = []
#     dir = os.path.expanduser(dir)
#     for target in sorted(os.listdir(dir)):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue
#
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in sorted(fnames):
#                 if has_file_allowed_extension(fname, extensions):
#                     path = os.path.join(root, fname)
#                     item = (path, class_to_idx[target])
#                     images.append(item)
#
#     return images

import random

class SampleLoader(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, categories, categories_counts, root, loader=default_loader, transform=None, target_transform=None, sample_size=1000000):
        # classes, class_to_idx = find_classes(root)
        # samples = make_dataset(root, class_to_idx, extensions)


        self.root = root
        self.loader = loader
        # self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.n_samples = sample_size

        self.category_samples = categories
        self.category_sample_counts = categories_counts
        self.n_categories = len(categories)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        category_id = random.randint(0, self.n_categories-1)
        category_count = self.category_sample_counts[category_id]
        instance_id = random.randint(0, category_count-1)
        rel_path = self.category_samples[category_id][instance_id]
        path = os.path.join(self.root, rel_path)
        target = category_id

        # path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            print("WARN: {} Problematic!, Skip!".format(path))

            return None # One solution
            # print("WARN: {} Problematic!".format(path))
            # instance_id = random.randint(0, category_count - 1)
            # rel_path = self.category_samples[category_id][instance_id]
            # path = os.path.join(self.root, rel_path)
            #
            # sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.n_samples

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Sample 0 Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str








