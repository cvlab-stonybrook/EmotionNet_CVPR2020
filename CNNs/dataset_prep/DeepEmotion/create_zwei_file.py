# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): based on train/test files provided by Rameswar, set it to my costumized directories
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 17:34


# TODO: this script is more or less one-time thing, no need to make it fancy!

import os
from PyUtils.pickle_utils import save2pickle
file_split = 'train'  # candidate: train_sample (1:1), test
src_file_path = '/home/zwei/datasets/emotion_datasets/deepemotion/index_files/{}.txt'.format(file_split)

positive_set = (1, 4, 6, 7)
negative_set = []

def convert6_2(label):
        if int(label) in [1, 4, 6, 7]:
            label = 0  # NEGATIVE
        else:
            label = 1  # POSITIVE
        return label



root_file = '/home/zwei/datasets/emotion_datasets/deepemotion'
data_set_information = []
counts = [0, 0]

with open(src_file_path, 'r') as if_:
    for line in if_:
        contents = line.strip().split(' ')
        abs_image_path = contents[0]
        image_broad_categories = contents[1]
        new_label = convert6_2(image_broad_categories)
        counts[new_label] += 1
        abs_image_path_parts = abs_image_path.split(os.sep)
        rel_image_path = os.path.join(abs_image_path_parts[-2], abs_image_path_parts[-1])
        if os.path.exists(os.path.join(root_file, rel_image_path)):
            data_set_information.append((rel_image_path, new_label))

        else:
            print("{:s} Not Exist".format(rel_image_path))

save2pickle(os.path.join(root_file, '{0}.pkl'.format(file_split)), data_set_information)

print("DEB")






