# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): based on train/test files provided by Rameswar, set it to my costumized directories
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 17:34


# TODO: this script is more or less one-time thing, no need to make it fancy!

import os
from PyUtils.pickle_utils import save2pickle
file_split = 'test'  # candidate: train_sample (1:1), test
src_file_path = '/home/zwei/datasets/emotion_datasets/finegrained_emotion_google/{}.txt'.format(file_split)


negative_labels = ['anger', 'fear', 'sadness']
positive_labels = ['joy', 'love']

def convert6_2(label):
        if label in negative_labels:
            label = 0  # NEGATIVE
        elif label in positive_labels:
            label = 1  # POSITIVE
        else:
            label = None
        return  label



root_file = '/home/zwei/datasets/emotion_datasets/finegrained_emotion_google'
data_set_information = []
counts = [0, 0]

with open(src_file_path, 'r') as if_:
    for line in if_:
        contents = line.strip().split(' ')
        if len(contents)>2:
            contents[0] = ' '.join(contents[:-1])
        abs_image_path = contents[0]
        abs_image_path_parts = abs_image_path.split(os.sep)

        image_broad_categories = abs_image_path_parts[-3]
        new_label = convert6_2(image_broad_categories)
        if new_label is None:
            continue

        counts[new_label] += 1
        rel_image_path = os.path.join(*abs_image_path_parts[-4:])
        if os.path.exists(os.path.join(root_file, rel_image_path)):
            data_set_information.append((rel_image_path, new_label))

        else:
            print("{:s} Not Exist".format(rel_image_path))

save2pickle(os.path.join(root_file, '{0}-wo-surprise.pkl'.format(file_split)), data_set_information)

print("DEB")






