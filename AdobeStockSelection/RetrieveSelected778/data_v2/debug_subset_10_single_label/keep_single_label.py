# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 18/Feb/2019 13:56

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import get_key_sorted_dict
split = 'test'
data_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/CNNsplit_{}.pkl'.format(split))
single_label_list = []
label_count_dict = {}

for s_data in data_list:
    if len(s_data[1]) == 1:
        single_label_list.append(s_data)
        s_label = s_data[1][0]
        if s_label in label_count_dict:
            label_count_dict[s_label] += 1
        else:
            label_count_dict[s_label] = 1

total = 0
label_count_dict = get_key_sorted_dict(label_count_dict)
for s_label in label_count_dict:
    print("{}\t{}".format(s_label, label_count_dict[s_label]))
    total += label_count_dict[s_label]

print("total: {}".format(total))

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10_single_label/CNNsplit_{}.pkl'.format(split), single_label_list)