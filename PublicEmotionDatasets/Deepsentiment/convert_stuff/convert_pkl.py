# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 16:52
import os
from PyUtils.pickle_utils import save2pickle
import tqdm


user_root = os.path.expanduser('~')
dataset_dir = os.path.join(user_root, 'datasets/PublicEmotion', 'Deepsentiment')
data_split = 'train_3'

split_file = os.path.join(dataset_dir, '{}_agree.txt'.format(data_split))

data_set = []
data_counts = {}
with open(split_file, 'r') as of_:
    lines = of_.readlines()
    for s_line in tqdm.tqdm(lines):
        s_parts = s_line.strip().split(' ')
        s_file = s_parts[0]
        s_label = int(s_parts[1])
        data_set.append([s_file, s_label])
        # if s_label == 1:
        if s_label in data_counts:
            data_counts[s_label] += 1
        else:
            data_counts[s_label] = 1
print("total: {}".format(len(data_set)))
print("{}".format( ', '.join('{}\t{}'.format(x, data_counts[x]) for x in data_counts)))

save2pickle(os.path.join(dataset_dir, 'z_data', '{}.pkl'.format(data_split)), data_set)