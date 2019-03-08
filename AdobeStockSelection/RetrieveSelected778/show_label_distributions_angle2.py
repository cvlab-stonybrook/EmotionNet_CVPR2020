# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): showing results from the second angle
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 18:15
from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import get_value_sorted_dict
import tqdm
subset = 'train'

print(subset)
train_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_{}.pkl'.format(subset))
dataset_keyword_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_keyword_annotations.pkl')
train_set = set(train_list)
assert len(train_set) == len(train_list)

tag_image_count = {}
for s_image_cid in tqdm.tqdm(dataset_keyword_annotations, total=len(dataset_keyword_annotations)):
    if s_image_cid not in train_set:
        continue
    s_image_annotations = dataset_keyword_annotations[s_image_cid]
    for s_tag in s_image_annotations:
        if s_tag in tag_image_count:
            tag_image_count[s_tag] += 1
        else:
            tag_image_count[s_tag] = 1



tag_image_count = get_value_sorted_dict(tag_image_count, reverse=False)
for s_tag in tag_image_count:
    print("{}\t{}".format(s_tag, tag_image_count[s_tag]))