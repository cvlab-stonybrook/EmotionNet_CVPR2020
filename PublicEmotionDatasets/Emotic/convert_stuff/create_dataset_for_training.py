# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 08/Mar/2019 09:25

from PyUtils.pickle_utils import loadpickle, save2pickle
import os, glob, tqdm
from PublicEmotionDatasets.Emotic.constants import emorion2idx
data_split = 'test'
annotaiton_data = loadpickle('/home/zwei/datasets/PublicEmotion/EMOTIC/new_z_data/{}_whole_image.pkl'.format(data_split))


data_list = []
for s_annotation in tqdm.tqdm(annotaiton_data):
    s_emotion_counter = s_annotation[1]
    s_label = []
    for s_item in s_emotion_counter.most_common():
        s_label.append([emorion2idx[s_item[0]], s_item[1]])
    data_list.append([s_annotation[0], s_label])

save2pickle('/home/zwei/datasets/PublicEmotion/EMOTIC/new_z_data/{}_image_based_idx+count_forpytorch.pkl'.format(data_split), data_list)

print("DB")