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




class ImageTextDatasetList(data.Dataset):
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
        path = self.samples[index][0]
        label = self.samples[index][1]
        text = self.samples[index][2]

        try:
            sample = self.loader(path)
        except:
            print("WARN: {} Problematic!, Skip!".format(path))

            return None

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            # target = self.target_transform(target)
            # new_targets = []
            # for s_target, s_transform in zip(target, self.target_transform):
            #     new_targets.append(s_transform(s_target))
            label = self.target_transform[0](label)
            text = self.target_transform[1](text)

        return sample, label, text

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





class ImageTextRelLists(ImageTextDatasetList):
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
            samples.append((image_full_path, *s_item[1:]))

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + image_root))

        super(ImageTextRelLists, self).__init__(samples, loader,
                                                transform=transform,
                                                target_transform=target_transform)
        # self.imgs = self.samples







