# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 17:21

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import get_value_sorted_dict
import random
import os
from PyUtils.json_utils import load_json_list


tag_frequencies = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')
banned_cid_set = set(loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/cids_banned_for_similar_tags.pkl'))

collected_CIDs = set()
max_selection=100
for s_idx, s_tag in enumerate(tag_frequencies):
    added_count = 0
    repeated_count = 0
    s_cids = tag_frequencies[s_tag]

    for s_CID in s_cids:
        if s_CID in banned_cid_set:
            continue
        if s_CID not in collected_CIDs:
            collected_CIDs.add(s_CID)
            added_count += 1
            if added_count >= max_selection:
                break
        else:
            repeated_count+=1
    print("{}\t {},\t Repeated:{}\t, Added: {}, total: {}".format(s_idx, s_tag, repeated_count, added_count, len(s_cids)))

save2pickle('data_v2/dataset_CIDs_742_test.pkl', list(collected_CIDs))
print("collected: {}".format(len(collected_CIDs)))