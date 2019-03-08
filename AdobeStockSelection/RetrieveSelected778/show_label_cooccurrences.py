# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): showing results from the second angle
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 18:15
from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import get_value_sorted_dict, string_list2dict
import tqdm
import numpy
import itertools
subset = 'train'

print(subset)
train_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_{}.pkl'.format(subset))
dataset_keyword_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_keyword_annotations.pkl')
train_set = set(train_list)
assert len(train_set) == len(train_list)

tag_combo_count = {}
for s_image_cid in tqdm.tqdm(dataset_keyword_annotations, total=len(dataset_keyword_annotations)):
    if s_image_cid not in train_set:
        continue
    s_image_annotations = dataset_keyword_annotations[s_image_cid]
    if len(s_image_annotations) < 2:
        continue

    else:
        tag_combinations = itertools.combinations(s_image_annotations, 2)
        for s_tag_combination in tag_combinations:
            s_string = '-'.join(s_tag_combination)
            if s_string in tag_combo_count:
                tag_combo_count[s_string] += 1
            else:
                s_string = '-'.join(s_tag_combination[::-1])
                if s_string in tag_combo_count:
                    tag_combo_count[s_string] += 1
                else:
                    tag_combo_count[s_string] = 1





tag_combo_count = get_value_sorted_dict(tag_combo_count, reverse=False)
for s_tag in tag_combo_count:
    print("{}\t{}".format(s_tag, tag_combo_count[s_tag]))