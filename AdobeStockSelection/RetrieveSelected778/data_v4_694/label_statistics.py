# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 20/Feb/2019 09:26

from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import get_value_sorted_dict
data_split = 'test'
image_annotations_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v4_694/CNNsplit_{}_dict.pkl'.format(data_split))
image_urls = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_image_urls.pkl')
vocabularies = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v4_694/tag-idx-conversion.pkl')
key2idx = vocabularies['key2idx']
idx2key = vocabularies['idx2key']
keyid_frequency_count = {}
total_counts = 0
for s_image_cid in image_annotations_dict:
    s_annotation = image_annotations_dict[s_image_cid]
    total_counts += len(s_annotation[1])
    for s_key_id in s_annotation[1]:
        if s_key_id in keyid_frequency_count:
            keyid_frequency_count[s_key_id] += 1
        else:
            keyid_frequency_count[s_key_id] = 1


keyid_frequency_count = get_value_sorted_dict(keyid_frequency_count)
average_tags = total_counts * 1. / len(image_annotations_dict)
for s_key_id in keyid_frequency_count:
    print("{}\t{}\t{}".format(idx2key[s_key_id], s_key_id, keyid_frequency_count[s_key_id]))
print("DB")