# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 17:16


# from .lsun import LSUN, LSUNClass
from .torchvision_raw import ImageFolder, DatasetFolder # This is directly copying from pytorch's implementation
from .basic_loader import ImageRelLists
from .imagename_loader import ImageNamesRelLists
# from .coco import CocoCaptions, CocoDetection
# from .cifar import CIFAR10, CIFAR100
# from .stl10 import STL10
# from .mnist import MNIST, EMNIST, FashionMNIST
# from .svhn import SVHN
# from .phototour import PhotoTour
# from .fakedata import FakeData
# from .semeion import SEMEION
# from .omniglot import Omniglot
#
# __all__ = ('LSUN', 'LSUNClass',
#            'ImageFolder', 'DatasetFolder', 'FakeData',
#            'CocoCaptions', 'CocoDetection',
#            'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST',
#            'MNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
#            'Omniglot')
