# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 18/Feb/2019 22:18

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import string_list2dict

previous_dict_info = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag-idx-conversion.pkl')
previous_idx2keyword = previous_dict_info['idx2key']
previous_keyword2idx = previous_dict_info['key2idx']

excluded_keyword_file = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/not_in_vocabulary_list.txt'
excluded_keywords = []
with open(excluded_keyword_file, 'r') as of_:
    text_lines = of_.readlines()
    for s_line in text_lines:
        s_parts = s_line.strip().split(' ')
        excluded_keywords.append(s_parts[0])

excluded_keywords_set = set(excluded_keywords)
new_keywords_list = []
for s_word in previous_keyword2idx:
    if s_word in excluded_keywords_set:
        continue
    else:
        new_keywords_list.append(s_word)

new_keywords_list = sorted(new_keywords_list)
idx2key, key2idx = string_list2dict(new_keywords_list)
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl', {'idx2key': idx2key, 'key2idx': key2idx})

print("DB")