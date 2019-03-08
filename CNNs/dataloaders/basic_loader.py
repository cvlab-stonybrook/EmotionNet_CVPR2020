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


class DatasetList(data.Dataset):
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

    def __init__(self, samples, loader, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        # samples = make_dataset(root, class_to_idx, extensions)


        # self.root = root
        self.loader = loader
        # self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            print("WARN: {} Problematic!, Skip!".format(path))

            return None

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Sample 0 Location: {}\n'.format(self.samples[0])
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str





class ImageRelLists(DatasetList):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, image_paths, image_root, transform=None, target_transform=None,
                 loader=default_loader):

        samples = []
        for s_item in image_paths:
            image_full_path = os.path.join(image_root, s_item[0])
            samples.append((image_full_path, s_item[1]))

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + image_root))

        super(ImageRelLists, self).__init__(samples, loader,
                                            transform=transform,
                                            target_transform=target_transform)
        # self.imgs = self.samples


class ImageReDicts(DatasetList):
    def __init__(self, image_ids, annotation_dicts, image_root, transform=None, target_transform=None,
                 loader=default_loader):

        samples = []
        for s_image_id in image_ids:
            s_information = annotation_dicts[s_image_id]
            image_full_path = os.path.join(image_root, s_information[0])
            samples.append((image_full_path, s_information[1:]))

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + image_root))

        super(ImageReDicts, self).__init__(samples, loader,
                                            transform=transform,
                                            target_transform=target_transform)
        self.imgs = self.samples


# class WebEmoRawDicts(DatasetList):
#     def __init__(self, image_ids, annotation_dicts, image_root, transform=None, target_transform=None,
#                  loader=default_loader):
#
#         samples = []
#         for s_image_id in image_ids:
#             s_information = annotation_dicts[s_image_id]
#             image_full_path = os.path.join(image_root, s_information[0])
#             samples.append((image_full_path, s_information[1:]))
#
#         if len(samples) == 0:
#             raise(RuntimeError("Found 0 files in subfolders of: " + image_root))
#
#         super(WebEmoRawDicts, self).__init__(samples, loader,
#                                             transform=transform,
#                                             target_transform=target_transform)
#         self.imgs = self.samples

