# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 25/Feb/2019 16:49

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob, os, tqdm
from PyUtils.json_utils import load_json_list
from AdobeStockTools.TagUtils import remove_hat
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
from PyUtils.dict_utils import get_value_sorted_dict
selected_list = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_1000x10x5_v3/selected_list_10000.pkl')
selected_dict = {}
for s_item in selected_list:
    s_cid = int(get_image_cid_from_url(s_item[0]))
    selected_dict[s_cid] = s_item


processed_cids = set()
created_annotations = []
keyword_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/E_vocabulary.pkl')
key2idx = keyword_vocabulary['key2idx']
idx2key = keyword_vocabulary['idx2key']
val_key_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_val.pkl')
tag_counts = {}

for s_annotation in tqdm.tqdm(val_key_annotations):
    s_cid = int(get_image_cid_from_url(s_annotation[0], location=1))
    if s_cid in selected_dict:
        updated_tags = []
        for s_tag_id in s_annotation[1]:
            updated_tags.append(idx2key[s_tag_id])
            if idx2key[s_tag_id] in tag_counts:
                tag_counts[idx2key[s_tag_id]] += 1
            else:
                tag_counts[idx2key[s_tag_id]] = 1


        created_annotations.append([selected_dict[s_cid][0], updated_tags])


tag_counts = get_value_sorted_dict(tag_counts, reverse=False)
save2pickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionRemovalData/selected_10k_key_annotations_list.pkl', created_annotations)
print("DONE")


