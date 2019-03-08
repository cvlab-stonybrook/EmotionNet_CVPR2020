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
train_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/CNNsplit_{}.pkl'.format(subset))

idx2key = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/tag-idx-conversion.pkl')['idx2key']

id_image_count = {}
for s_image_item in train_list :
    s_tag_ids = s_image_item[1]

    for s_tag_id in s_tag_ids:
        if s_tag_id in id_image_count:
            id_image_count[s_tag_id] += 1
        else:
            id_image_count[s_tag_id] = 1


id_image_count = get_value_sorted_dict(id_image_count, reverse=False)
for s_tag in id_image_count:
    print("{}\t{}\t{}".format(s_tag, idx2key[s_tag], id_image_count[s_tag]))