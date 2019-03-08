# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 21:59

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob
import os
import tqdm
from PyUtils.json_utils import load_json_list
from AdobeStockTools.TagUtils import remove_hat
from EmotionTag.load_csv_annotations import load_verified
train_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_train.pkl')
val_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_val.pkl')
test_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_test.pkl')

full_list = train_list + val_list + test_list
dataset_cid_set = set(full_list)
assert len(dataset_cid_set) == len(train_list) + len(val_list) + len(test_list)

cid_information = {}

tag_image_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')


for s_raw_idx, s_tag in enumerate(tqdm.tqdm(tag_image_dict, desc="Processing Files")):

    tag_cids = tag_image_dict[s_tag]

    for s_cid in tag_cids:
       if s_cid in dataset_cid_set:
           if s_cid in cid_information:
               cid_information[s_cid].append(s_tag)
           else:
               cid_information[s_cid] = [s_tag]


assert len(cid_information) == len(dataset_cid_set)

save2pickle('data_v2/dataset_keyword_annotations.pkl', cid_information)