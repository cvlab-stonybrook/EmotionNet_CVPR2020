# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 29/Oct/2018 10:45
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import get_value_sorted_dict
import tqdm
split = 'train'
raw_data = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/{}-CIDs-fullinfo.pkl'.format(split))

word_frequencies = {}
for s_data in tqdm.tqdm(raw_data):
    raw_tags = s_data[1]
    for s_tag in raw_tags:
        if s_tag in word_frequencies:
            word_frequencies[s_tag] += 1
        else:
            word_frequencies[s_tag] = 1

word_frequencies = get_value_sorted_dict(word_frequencies)
save2pickle('{}-word-frequencies.pkl'.format(split), word_frequencies)
print("DEB")