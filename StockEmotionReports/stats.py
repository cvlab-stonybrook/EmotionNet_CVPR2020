# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 15/Mar/2019 22:32

from PyUtils.pickle_utils import loadpickle
import tqdm
import os
train_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')

number_of_emotion_tags = 0
number_of_complete_tags = 0
total_images = 0

emotion_tags = []
complete_tags = []


for s_data in tqdm.tqdm(train_data):
    s_name = s_data[0]
    s_name_parts = s_name.split(os.sep)
    if len(s_name_parts) == 2:
        number_of_complete_tags += len(s_data[2])
        number_of_emotion_tags += len(s_data[1])
        total_images += 1
        emotion_tags.append(len(s_data[1]))
        complete_tags.append(len(s_data[2]))

with open('emotion_counts.txt', "w") as of_:
    for s_count in emotion_tags:
        of_.write('{}\n'.format(s_count))

with open('full_counts.txt', "w") as of_:
    for s_count in complete_tags:
        of_.write('{}\n'.format(s_count))


print("DB")
