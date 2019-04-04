# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 12/Mar/2019 16:31
import tqdm
from PyUtils.pickle_utils import loadpickle, save2pickle
from collections import Counter
from TextClassificationV2.data_loader.AMT import counter2singlelabel, emotion2idx
import random

mturk_data = loadpickle('/home/zwei/Dev/AttributeNet3/MturkCollectedData/data/mturk_annotations.pkl')

data_list = []
thres = 3

for s_image_cid in tqdm.tqdm(mturk_data, total=len(mturk_data)):
    s_data = mturk_data[s_image_cid]
    s_rel_path = s_data['rel-path']
    s_image_emotions = []
    for x in s_data['image_emotion']:
        s_image_emotions.extend(x)
    s_image_emotions = Counter(s_image_emotions)
    multi_labels = []
    for s_item in s_image_emotions.most_common():
        multi_labels.append([emotion2idx[s_item[0]], s_item[1]])

    # single_label = counter2singlelabel(s_image_emotions, emotion2idx, thres=thres)
    # if single_label is not None:
    data_list.append([s_rel_path, multi_labels])

split = len(data_list) // 10

random.seed(0)
random.shuffle(data_list)

train_split = data_list[split:]
val_split = data_list[:split]

save2pickle('/home/zwei/Dev/AttributeNet3/MturkCollectedData/data/train_mclss.pkl', train_split)

save2pickle('/home/zwei/Dev/AttributeNet3/MturkCollectedData/data/test_mclss.pkl', val_split)


print("DB")
