# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): Test using our data
# Email: hzwzijun@gmail.com
# Created: 14/Mar/2019 22:24


from TextClassificationV2.models.TextCNN import TextCNN_NLT as TextCNN
from TextClassificationV2.data_loader.AMT import read_AMT_complete_mtrain_mtest

import TextClassification.utils as utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
from sklearn.utils import shuffle
import numpy as np
import argparse
import copy
from PyUtils.pickle_utils import loadpickle
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import random
from torch.optim import lr_scheduler
import tqdm
from CNNs.utils.util import AverageMeter
from CNNs.utils.util import report_params
from argparse import Namespace
import shutil
from PyUtils.file_utils import get_file_dir

from TextClassificationV2.data_loader.AMT import idx2emotion

def create_data_args(data_splits):
    args_data = Namespace()
    args_data.vocab = sorted(list(set([w for x_split in data_splits for sent in x_split for w in sent[0]])))
    args_data.classes = [0, 1, 2, 3, 4, 5, 6, 7] # FIXME: do this in a better way!
    args_data.tag2idx = {w: i for i, w in enumerate(args_data.vocab)}
    args_data.idx2tag = {i: w for i, w in enumerate(args_data.vocab)}
    return args_data


def pad_sentences(x, max_len, pad_id):
    if len(x) >= max_len:
        return x[:max_len]
    else:
        x = x + [pad_id] * (max_len - len(x))
    return x

def main():
    text_ckpt = torch.load('/home/zwei/Dev/AttributeNet3/TextClassificationV2/ckpts/TextCNN_googlenews_NLT_Static.pth.tar')
    args_model = text_ckpt['args_model']
    args_data = text_ckpt['args_data']
    text_model = TextCNN(args_model)
    model_tag2idx = args_data.tag2idx
    text_model.load_state_dict(text_ckpt['state_dict'], strict=True)
    vocab_idx2tag = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/vocab_dict.pkl')['idx2tag']
    dataset = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')
    text_model.eval()
    emotion_tags = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/Emotion_vocabulary.pkl')['key2idx']

    image_url_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_image_urls.pkl')

    for data_idx, s_data in enumerate(dataset):
        if data_idx % 10000 != 0:
            continue
        x_tags = [vocab_idx2tag[x] for x in s_data[2]]
        x_tag_ids = []
        x_tag_names = []
        x_emotion_tags = []
        for x_tag in x_tags:
            if x_tag in model_tag2idx:
                x_tag_ids.append(model_tag2idx[x_tag])
                x_tag_names.append(x_tag)
                if x_tag in emotion_tags:
                    x_emotion_tags.append(x_tag)
        x_tag_ids = pad_sentences(x_tag_ids, args_model.max_len, args_model.vocab_size+1)
        x_tag_ids = torch.LongTensor(x_tag_ids).unsqueeze(0)
        predicts = F.softmax(text_model(x_tag_ids)[0], dim=1).squeeze(0).cpu().data.numpy()
        image_cid = int(get_image_cid_from_url(s_data[0], location=1))
        if image_cid in image_url_dict:
            print("{}".format(image_url_dict[image_cid]))
            print(", ".join(x_emotion_tags))
            print(", ".join(x_tag_names))
            print(', '.join('{}({:.2f})'.format(idx2emotion[i], predicts[i]) for i in range(len(predicts)) ))









if __name__ == "__main__":
    main()
