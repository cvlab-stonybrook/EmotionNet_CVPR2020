# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 20:24

from PyUtils.pickle_utils import loadpickle
import os
import tqdm
from PyUtils.dict_utils import string_list2dict,get_key_sorted_dict
data_split = 'train'
data = loadpickle(os.path.join('/home/zwei/datasets/PublicEmotion/Deepemotion/z_data', '{}_8.pkl'.format(data_split)))
emotion_categories = sorted(['fear', 'sadness', 'excitement', 'amusement', 'anger', 'awe', 'contentment', 'disgust'])
idx2emotion, emotion2idx = string_list2dict(emotion_categories)

data_counts_8 = {}
data_counts_2 = {}
for s_data in tqdm.tqdm(data):
    if s_data[1] in data_counts_8:
        data_counts_8[s_data[1]] += 1
    else:
        data_counts_8[s_data[1]] = 1

    if s_data[2] in data_counts_2:
        data_counts_2[s_data[2]] += 1
    else:
        data_counts_2[s_data[2]] = 1
data_counts_8 = get_key_sorted_dict(data_counts_8, reverse=False)
print("{}".format( ', '.join('{}\t{}'.format(idx2emotion[x], data_counts_8[x]) for x in data_counts_8)))
print("{}".format( ', '.join('{}\t{}'.format(x, data_counts_2[x]) for x in data_counts_2)))

print("Done")
