# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): compared to wo-surprise, this is classifying to 6 classes
# Email: hzwzijun@gmail.com
# Created: 10/Oct/2018 22:08


# TODO: this script is more or less one-time thing, no need to make it fancy!

import os
from PyUtils.pickle_utils import save2pickle
file_split = 'train'  # candidate: train_sample (1:1), test
src_file_path = '/home/zwei/datasets/emotion_datasets/finegrained_emotion_google/{}.txt'.format(file_split)


class2idx = {'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}


root_file = '/home/zwei/datasets/emotion_datasets/finegrained_emotion_google'
data_set_information = []
import numpy as np

counts = np.zeros(len(class2idx))

with open(src_file_path, 'r') as if_:
    for line in if_:
        contents = line.strip().split(' ')
        if len(contents)>2:
            contents[0] = ' '.join(contents[:-1])
        abs_image_path = contents[0]
        image_category = int(contents[-1])
        abs_image_path_parts = abs_image_path.split(os.sep)

        image_broad_categories = class2idx[abs_image_path_parts[-3]]
        assert image_broad_categories == image_category, 'CHK'
        # new_label = convert6_2(image_broad_categories)
        # if new_label is None:
        #     continue

        counts[image_broad_categories] += 1
        rel_image_path = os.path.join(*abs_image_path_parts[-4:])
        if os.path.exists(os.path.join(root_file, rel_image_path)):
            data_set_information.append((rel_image_path, image_broad_categories))

        else:
            print("{:s} Not Exist".format(rel_image_path))

save2pickle(os.path.join(root_file, '{0}-6.pkl'.format(file_split)), data_set_information)

print("DEB")






