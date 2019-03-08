# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 20:54

from PyUtils.pickle_utils import loadpickle

val_cid_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_test.pkl')
val_tag_set = set(val_cid_list)
tag_frequencies = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')
for s_tag in tag_frequencies:
    count = 0
    selected_print_cids = []
    for s_cid in tag_frequencies[s_tag]:
        if s_cid in val_tag_set:
            selected_print_cids.append(s_cid)
            count += 1
            if count >= 10:
                break

    print("{}\t{}".format(s_tag, ', '.join(str(x) for x in selected_print_cids)))
print("DB")