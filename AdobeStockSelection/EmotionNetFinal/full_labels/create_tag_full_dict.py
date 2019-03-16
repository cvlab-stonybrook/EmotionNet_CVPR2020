# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create tag label and tag set to train two branch
# Email: hzwzijun@gmail.com
# Created: 03/Mar/2019 09:14

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob, os, tqdm
from PyUtils.json_utils import load_json_list
from AdobeStockTools.TagUtils import remove_hat
import torch

cid_set = set(loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/full_labels/CID_list.pkl'))
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
processed_cids = set()

cid_full_tag_dict = {}

for s_raw_file in tqdm.tqdm(raw_annotation_files):
    raw_annotations = load_json_list(s_raw_file)
    for s_raw_annotation in raw_annotations:
        image_cid = s_raw_annotation['cid']
        if image_cid not in cid_set or image_cid in processed_cids:
            continue
        else:
            processed_cids.add(image_cid)

            raw_tags = remove_hat(s_raw_annotation['tags'])
            cid_full_tag_dict[image_cid] = raw_tags


print("total data {}".format(len(cid_full_tag_dict)))
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/full_labels/CID_full_tag_dict.pkl', cid_full_tag_dict)
