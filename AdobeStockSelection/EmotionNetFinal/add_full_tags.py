# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): Add full tag ids
# Email: hzwzijun@gmail.com
# Created: 13/Mar/2019 21:54


from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url

tag2idx = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/vocab_dict.pkl')['tag2idx']

cid_full_tag = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/full_labels/CID_full_tag_dict.pkl')

orig_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/tag_labels/CNNsplit_tag_labels_test.pkl')

full_tag_data = []

for s_data in tqdm.tqdm(orig_data):
    s_cid = int(get_image_cid_from_url(s_data[0], 1))
    s_full_tags = cid_full_tag[s_cid]
    s_full_tag_ids = []
    for s_tag in s_full_tags:
        if s_tag in tag2idx:
            s_full_tag_ids.append(tag2idx[s_tag])
        else:
            print("{} is not included".format(s_tag))
    full_tag_data.append([s_data[0], s_data[1], s_full_tag_ids])

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_test.pkl', full_tag_data)