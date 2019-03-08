# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Feb/2019 19:23

from PyUtils.pickle_utils import loadpickle
import tqdm

data_split = 'training'
v3_data_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/CNNsplit_{}_dict.pkl'.format(data_split))
v4_data_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v4_694/CNNsplit_{}_dict.pkl'.format(data_split))
image_urls = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_image_urls.pkl')
vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl')
assert len(v3_data_dict) == len(v4_data_dict)
key2idx = vocabulary['key2idx']
idx2key = vocabulary['idx2key']

for image_idx, s_image_cid in enumerate(v3_data_dict):
    s_image_v3_data = v3_data_dict[s_image_cid]
    s_image_v4_data = v4_data_dict[s_image_cid]
    s_image_v3_keywords = s_image_v3_data[1]
    s_image_v4_keywords = s_image_v4_data[1]

    common_keywords = set(s_image_v3_keywords).intersection(set(s_image_v4_keywords))
    print("{}\t{}\t{}".format(image_idx, s_image_cid, image_urls[s_image_cid]))
    print("keywords: {}".format(', '.join(['{}({})'.format(idx2key[x], x) for x in s_image_v3_keywords])))
    print("tags: {}".format(', '.join(['{}({})'.format(idx2key[x], x) for x in s_image_v4_keywords])))
    # v3_only_keys = []
    # for s_key in s_image_v3_keywords:
    #
    #     if s_key not in common_keywords:
    #         v3_only_keys.append(s_key)
    #
    # print("{}: {} is only in v3".format(s_key, idx2key[s_key]))
    #
    # for s_key in s_image_v4_keywords:
    #     if s_key not in common_keywords:
    #         print("{}: {} is only in v4".format(s_key, idx2key[s_key]))

