# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 18/Feb/2019 23:01

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
from PyUtils.dict_utils import get_key_sorted_dict, get_value_sorted_dict
split = 'test'

previous_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/CNNsplit_{}.pkl'.format(split))
previous_idx2key = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag-idx-conversion.pkl')['idx2key']

current_key2idx = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl')['key2idx']
current_data = []

current_category_counts = {}
for s_data in tqdm.tqdm(previous_data):
    s_keywords = s_data[1]
    new_keywords = []
    for s_keyword in s_keywords:
        if previous_idx2key[s_keyword] in current_key2idx:
            current_idx = current_key2idx[previous_idx2key[s_keyword]]
            if current_idx in current_category_counts:
                current_category_counts[current_idx] += 1
            else:
                current_category_counts[current_idx] = 1
            new_keywords.append(current_idx)
    if len(new_keywords) < 1:
        continue
    else:
        current_data.append([s_data[0], new_keywords])

current_category_counts = get_key_sorted_dict(current_category_counts, reverse=False)

total = 0
for s_idx in current_category_counts:
    print('{}\t{}'.format(s_idx, current_category_counts[s_idx]))
    total += current_category_counts[s_idx]
print("Total tags: {}\t total images: {}".format(total, len(current_data)))

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/CNNsplit_{}.pkl'.format(split), current_data)