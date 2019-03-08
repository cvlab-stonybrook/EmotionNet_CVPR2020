# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 18:15
from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import get_value_sorted_dict
subset = 'train'
print(subset)
train_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_{}.pkl'.format(subset))
tag_cid_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')

train_set = set(train_list)
assert len(train_set) == len(train_list)

tag_image_count = {}
for s_tag in tag_cid_dict:
    s_tag_cids = tag_cid_dict[s_tag]
    s_tag_image_count = 0
    for s_cid in s_tag_cids:
        if s_cid in train_set:
            s_tag_image_count += 1

    tag_image_count[s_tag] = s_tag_image_count
tag_image_count = get_value_sorted_dict(tag_image_count, reverse=False)
for s_tag in tag_image_count:
    print("{}\t{}".format(s_tag, tag_image_count[s_tag]))