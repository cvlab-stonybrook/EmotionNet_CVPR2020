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

annotation_directory = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2'
raw_annotation_files = glob.glob(os.path.join(annotation_directory, '*.json'))
processedCIDs = set()
image_cid_urls = {}
for s_raw_annotation_file in tqdm.tqdm(raw_annotation_files, 'Processing Files'):
    s_annotation_list = load_json_list(s_raw_annotation_file)
    for s_annotation in s_annotation_list:
        s_cid = s_annotation['cid']
        if s_cid in processedCIDs or s_cid not in dataset_cid_set:
            continue
        else:
            processedCIDs.add(s_cid)
            assert s_cid not in image_cid_urls
            image_cid_urls[s_cid] = s_annotation['url']

assert len(image_cid_urls) == len(dataset_cid_set)

save2pickle('data_v2/dataset_image_urls.pkl', image_cid_urls)