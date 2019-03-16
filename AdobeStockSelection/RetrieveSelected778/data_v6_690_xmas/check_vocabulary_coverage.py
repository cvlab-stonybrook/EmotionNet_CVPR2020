# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 23/Feb/2019 15:12

from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import get_value_sorted_dict
tag_counts = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/train_tag_counts.pkl')

selected_tags = {}
discarded_tags = {}
for s_tag in tag_counts:
    if tag_counts[s_tag] <= 100:
        discarded_tags[s_tag] = tag_counts[s_tag]
    else:
        selected_tags[s_tag] = tag_counts[s_tag]
print("DB")
emotion_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/Emotion_vocabulary.pkl')
emotion2idx=  emotion_vocabulary['key2idx']
for s_key in emotion2idx:
    if s_key not in selected_tags:
        print("{} has low or no frequenct".format(s_key))
    else:
        print('{}\t{}'.format(s_key, tag_counts[s_key]))