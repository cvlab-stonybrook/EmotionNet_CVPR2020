# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 20:54

import os
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import string_list2dict, get_key_sorted_dict
import tqdm
from PublicEmotionDatasets.Unbiasedemotion.constants import emotion2idx, idx2emotion
user_root = os.path.expanduser('~')
dataset_dir = os.path.join(user_root, 'datasets/PublicEmotion', 'UnBiasedEmo')
z_data_dir = os.path.join(dataset_dir, 'z_data')

data_split = 'test-wo-surprise'
dataset = loadpickle(os.path.join(dataset_dir, 'previous_annotations', '{}.pkl'.format(data_split)))
new_dataset = []
label_counts = {}
for s_data in tqdm.tqdm(dataset):
    s_data_rel_path = s_data[0]
    s_data_cur_path = os.path.join(dataset_dir, 'images', *(s_data_rel_path.split(os.sep)[1:]))
    if os.path.exists(s_data_cur_path):

        new_dataset.append([os.path.join(*(s_data_rel_path.split(os.sep)[1:])), s_data[1]])
        if s_data[1] in label_counts:
            label_counts[s_data[1]] += 1
        else:
            label_counts[s_data[1]] = 1
    else:
        print("{} Not Exist".format(s_data[0]))

label_counts = get_key_sorted_dict(label_counts, reverse=False)
print("{}".format( ', '.join('{}\t{}'.format(idx2emotion[x], label_counts[x]) for x in label_counts)))

save2pickle(os.path.join(z_data_dir, '{}.pkl'.format(data_split)), new_dataset)
print("DB")