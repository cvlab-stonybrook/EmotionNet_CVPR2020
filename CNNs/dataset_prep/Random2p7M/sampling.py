# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Oct/2018 11:42

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.file_utils import get_stem
import os
import numpy as np
import tqdm

data_file = '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/Random2p7M/data/Emotion6-NN4.pkl'
save_directory = os.path.dirname(data_file)
stem_name = get_stem(data_file)

data = loadpickle(data_file)

annotations = data['annotations']
Counts = data['counts']

trainCategories = {}
trainCounts = {}

for s_item in tqdm.tqdm(annotations):
    s_class = s_item[1]
    if s_class not in trainCategories:
        trainCategories[s_class] = [s_item[0]]
        trainCounts[s_class] = 1
    else:
        trainCategories[s_class].append(s_item[0])
        trainCounts[s_class] += 1

# for s_category in trainCounts:
#     assert trainCounts[s_category] == Counts[s_category]



traindata = {'categories': trainCategories, 'counts': trainCounts}
save2pickle(os.path.join(save_directory, '{}_sampling.pkl'.format(stem_name)), traindata)









