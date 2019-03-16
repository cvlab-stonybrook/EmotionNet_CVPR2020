# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create tag labels for train/val/test from their tag set
# Email: hzwzijun@gmail.com
# Created: 25/Feb/2019 13:47
import tqdm
import os, glob
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.json_utils import load_json_list
from AdobeStockTools.TagUtils import remove_hat
from PyUtils.dict_utils import get_value_sorted_dict
data_split = 'test'

data_key_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_{}_dict.pkl'.format(data_split))
data_tag_dict = {}
keyword_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/E_vocabulary.pkl')
key2idx = keyword_vocabulary['key2idx']
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))

tag_counts = {}
for s_raw_annotation_file in tqdm.tqdm(raw_annotation_files):
    s_raw_annotations = load_json_list(s_raw_annotation_file)
    for s_raw_annotation in s_raw_annotations:
        s_cid = s_raw_annotation['cid']
        if s_cid in data_tag_dict or s_cid not in data_key_dict:
            continue
        else:

            s_tags = remove_hat(s_raw_annotation['tags'])
            updated_tags = []
            for s_tag in s_tags:
                if s_tag in key2idx:
                    if s_tag in tag_counts:
                        tag_counts[s_tag] += 1
                    else:
                        tag_counts[s_tag] = 1

                    updated_tags.append(key2idx[s_tag])
            if len(updated_tags) > 0:

                data_tag_dict[s_cid] = [data_key_dict[s_cid][0], updated_tags]
            else:
                print("{} has no tags in 690 categories".format(s_cid))
print("Tag counts total {}".format(len(tag_counts)))
tag_counts = get_value_sorted_dict(tag_counts)
for idx, s_tag in enumerate(tag_counts):
    print("{}\t{}\t{}".format(idx, s_tag, tag_counts[s_tag]))



save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/tag_labels/CNNsplit_tag_labels_{}_dict.pkl'.format(data_split), data_tag_dict)



