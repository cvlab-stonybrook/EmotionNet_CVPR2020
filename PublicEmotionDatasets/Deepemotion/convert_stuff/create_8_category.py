# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 16:19

import os
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import string_list2dict
import tqdm

user_root = os.path.expanduser('~')
dataset_dir = os.path.join(user_root, 'datasets/PublicEmotion', 'Deepemotion')
z_data_dir = os.path.join(dataset_dir, 'z_data')
emotion_categories = sorted(['fear', 'sadness', 'excitement', 'amusement', 'anger', 'awe', 'contentment', 'disgust'])
idx2emotion, emotion2idx = string_list2dict(emotion_categories)
data_split = 'train_sample'
dataset = loadpickle(os.path.join(dataset_dir, '{}.pkl'.format(data_split)))
dataset_8 = []
for s_data in tqdm.tqdm(dataset):
    s_data_category = os.path.dirname(s_data[0])
    emotion_idx = emotion2idx[s_data_category]
    dataset_8.append([s_data[0], emotion_idx, s_data[1]])

save2pickle(os.path.join(z_data_dir, '{}_8.pkl'.format(data_split)), dataset_8)
print("DB")
# train = loadpickle(os.path.join(dataset_dir, 'train.pkl'))
# train_sample = loadpickle(os.path.join(dataset_dir, 'train_sample.pkl'))
# test = loadpickle(os.path.join(dataset_dir, 'test.pkl'))