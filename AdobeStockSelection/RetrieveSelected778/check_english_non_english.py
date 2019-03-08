# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 13/Feb/2019 00:52
import os
import glob
from PyUtils.json_utils import load_json_list
from nltk.corpus import wordnet
from AdobeStockTools.TagUtils import remove_hat
english_tag_file_list = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
previous_tag_file_list = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve', '*.json'))

for s_english_file in english_tag_file_list:
    s_filename = os.path.basename(s_english_file)
    s_non_english_file = os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve', s_filename)
    s_english_tag_list = load_json_list(s_english_file)
    s_non_english_tag_list = load_json_list(s_non_english_file)
    s_non_english_cids = set()
    for s_non_english_annotation in s_non_english_tag_list:
        s_non_english_cids.add(s_non_english_annotation['cid'])
    for s_annotation in s_english_tag_list:
        s_cid = s_annotation['cid']
        if s_cid not in s_non_english_cids:
            print("{} not found".format(s_cid))
        s_tags = remove_hat(s_annotation['tags'])
        bad_tags = []
        for s_tag in s_tags:
            if len(wordnet.synsets(s_tag))<1:
                bad_tags.append(s_tag)

        print(", ".join(bad_tags))
    print("DB")