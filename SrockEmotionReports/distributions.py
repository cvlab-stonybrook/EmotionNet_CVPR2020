# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 15/Mar/2019 22:32

from PyUtils.pickle_utils import loadpickle
import tqdm
import os
from PyUtils.dict_utils import get_value_sorted_dict
train_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')
emotion_dicts = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/Emotion_vocabulary.pkl')
idx2emotion = emotion_dicts['idx2key']
number_of_emotion_tags = 0
number_of_complete_tags = 0
total_images = 0

emotion_idx_counts = {}
for s_data in tqdm.tqdm(train_data):
    s_name = s_data[0]
    s_name_parts = s_name.split(os.sep)
    if len(s_name_parts) == 2:
        for s_emotion_idx in s_data[1]:
            if s_emotion_idx in emotion_idx_counts:
                emotion_idx_counts[s_emotion_idx] += 1
            else:
                emotion_idx_counts[s_emotion_idx] = 1

emotion_name_counts = {idx2emotion[key]: emotion_idx_counts[key] for key in emotion_idx_counts}
emotion_name_counts = get_value_sorted_dict(emotion_name_counts, reverse=True)

        # number_of_complete_tags += len(s_data[2])
        # number_of_emotion_tags += len(s_data[1])
        # total_images += 1


print("DB")
