# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 09/Oct/2018 10:22
import torchvision.transforms as transforms


def get_ResNet_normalization():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    return normalize


ResNet_normalize = get_ResNet_normalization()

def get_val_simple_transform(original_size=256, current_size=224):

    return transforms.Compose([
                transforms.Resize(original_size),
                transforms.CenterCrop(current_size),
                transforms.ToTensor(),
                ResNet_normalize,
            ])


# def get_val_256_transform(current_size=224):
#     return transforms.Compose([
#         # transforms.Resize(original_size),
#         transforms.CenterCrop(current_size),
#         transforms.ToTensor(),
#         ResNet_normalize,
#     ])


def get_train_simple_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ResNet_normalize,
    ])


