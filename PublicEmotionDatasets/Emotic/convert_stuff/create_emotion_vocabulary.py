# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 08/Mar/2019 09:10

from PyUtils.pickle_utils import loadpickle
import os, glob, tqdm

data_split = 'train'
annotaiton_data = loadpickle('/home/zwei/datasets/PublicEmotion/EMOTIC/z_data/{}_person_based.pkl'.format(data_split))

emotion_dictionary = set()
for s_annotation in tqdm.tqdm(annotaiton_data):
    s_emotion_counter = s_annotation[1]
    for s_item in s_emotion_counter.most_common():
        if s_item[0] in emotion_dictionary:
            continue
        else:
            emotion_dictionary.add(s_item[0])

emotion_dictionary = sorted(list(emotion_dictionary))


print("DB")