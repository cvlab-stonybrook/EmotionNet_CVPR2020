# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 24/Oct/2018 21:47


import os
import torch
from CNNs.dataloaders.basic_loader import ImageRelLists
from CNNs.dataloaders.transformations import *
from PyUtils.pickle_utils import loadpickle
import numpy as np


def simple_multitrans():
    def target_transform(label_list):

        return torch.FloatTensor(np.array(label_list))
    return target_transform

def multilabel2multihot(n_classes=500):
    def target_transform(label_list):
        label_vector = np.zeros(n_classes, dtype=np.float)
        for s_label in label_list:
            label_vector[s_label] = 1
        label_vector /= (len(label_list)*1.0)
        return torch.FloatTensor(label_vector)
    return target_transform

def multilabel2multi1(n_classes=500):
    def target_transform(label_list):
        label_vector = np.zeros(n_classes, dtype=np.float)
        for s_label in label_list:
            label_vector[s_label] = 1
        # label_vector /= len(label_list)*1.0
        return torch.FloatTensor(label_vector)
    return target_transform


def multilabelidxcount2multihot(n_classes=500):
    def target_transform(label_list):
        label_vector = np.zeros(n_classes, dtype=np.float)
        for s_label in label_list:
            label_vector[s_label[0]] = s_label[1]
        if np.sum(label_vector) == 0:
            label_vector = label_vector
        else:
            label_vector /= np.sum(label_vector)*1.0
        return torch.FloatTensor(label_vector)
    return target_transform




def multilabel_idxcount_v2_train(args):
    image_information = loadpickle(args.train_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=multilabelidxcount2multihot(args.num_classes))
    return dataset

def multilabel_idxcount_v2_val(args):
    image_information = loadpickle(args.val_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_val_simple_transform(), target_transform=multilabelidxcount2multihot(args.num_classes))
    return dataset


def multilabel_v2_train(args):
    #FIXME:
    # annotation_file = annotation_file.format('train')
    image_information = loadpickle(args.train_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=multilabel2multihot(args.num_classes))
    return dataset




def multilabel_v2_val(args):
    #FIXME:
    # annotation_file = annotation_file.format('val')
    image_information = loadpickle(args.val_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_val_simple_transform(), target_transform=multilabel2multihot(args.num_classes))
    return dataset


def simple_multilabel_train(args):
    #FIXME:
    # annotation_file = annotation_file.format('train')
    image_information = loadpickle(args.train_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=simple_multitrans())
    return dataset

def simple_multilabel_val(args):
    #FIXME:
    # annotation_file = annotation_file.format('train')
    image_information = loadpickle(args.val_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_val_simple_transform(), target_transform=simple_multitrans())
    return dataset


def singlelabel_v2_train(args):
    #FIXME:
    image_information = loadpickle(args.train_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=None)
    return dataset


def singlelabel_v2_val(args):
    #FIXME:
    image_information = loadpickle(args.val_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_val_simple_transform(), target_transform=None)
    return dataset


#
# def multilabel_test(args, annotation_file, data_dir):
#     #FIXME:
#     annotation_file = annotation_file.format('test')
#     image_information = loadpickle(annotation_file)
#     dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=multilabel2multihot(args.num_classes))
#     return dataset




def multilabel_train(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('train')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_train_simple_transform(), target_transform=multilabel2multihot(args.num_classes))
    return dataset


def multilabel_val(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('val')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=multilabel2multihot(args.num_classes))
    return dataset

def multilabel_test(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('test')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=multilabel2multihot(args.num_classes))
    return dataset

def multilabel_BCE_train(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('train')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_train_simple_transform(), target_transform=multilabel2multi1(args.num_classes))
    return dataset


def multilabel_BCE_val(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('val')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=multilabel2multi1(args.num_classes))
    return dataset

def multilabel_BCE_test(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('test')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=multilabel2multi1(args.num_classes))
    return dataset


def singlelabel_train(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('train')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_train_simple_transform(), target_transform=None)
    return dataset


def singlelabel_val(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('val')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=None)
    return dataset

def singlelabel_test(args, annotation_file, data_dir):
    #FIXME:
    annotation_file = annotation_file.format('test')
    image_information = loadpickle(annotation_file)
    dataset = ImageRelLists(image_paths=image_information, image_root=data_dir, transform=get_val_simple_transform(), target_transform=None)
    return dataset



if __name__ == '__main__':
    # x_transform = multilabel2multihot(500)
    # x = x_transform([4, 10])
    # print("DEB")
    from argparse import Namespace
    from CNNs.dataloaders.utils import none_collate

    args = Namespace(num_classes=742)
    annotation_file = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/CNNsplit_{}.pkl'
    data_dir = '/home/zwei/datasets/stockimage_742/images-256'
    dataset = multilabel_val(args, annotation_file, data_dir)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=10, shuffle=False,
                                             num_workers=4, pin_memory=True, collate_fn=none_collate)
    import tqdm

    for s_images, s_labels in tqdm.tqdm(val_loader):
        pass

    print("Done")