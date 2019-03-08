# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create the vocabulary
# Email: hzwzijun@gmail.com
# Created: 15/Feb/2019 12:46

import glob
import os
import tqdm
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.json_utils import load_json_list
from PyUtils.dict_utils import string_list2dict
from EmotionTag.load_csv_annotations import load_verified
from nltk.corpus import wordnet
from AdobeStockTools.TagUtils import remove_hat, has_digits
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))

predefined_vocabularies = set(loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl').keys())

valid_annotation_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/CNNsplit_train.pkl')
train_cid_list = []
for s_item in tqdm.tqdm(valid_annotation_list, desc="Processing image cids"):
    train_cid_list.append(int(get_image_cid_from_url(s_item[0], location=1)))

train_cid_set = set(train_cid_list)

processedCIDs = set()
vocabularies = set()
bad_vocabularies = set()
for s_file in tqdm.tqdm(raw_annotation_files):
    keyword_raw_annotations = load_json_list(s_file)
    for s_annotation in keyword_raw_annotations:
        s_cid = s_annotation['cid']
        if s_cid not in train_cid_set or s_cid in processedCIDs:
            continue
        processedCIDs.add(s_cid)
        s_tags = remove_hat(s_annotation['tags'])

        for s_tag in s_tags:
            s_tag = s_tag.lower()
            if s_tag in vocabularies or s_tag in bad_vocabularies:
                continue
            else:
                if s_tag in predefined_vocabularies:
                    vocabularies.add(s_tag)
                elif len(wordnet.synsets(s_tag))>=1:
                    if len(s_tag) < 3 or has_digits(s_tag):
                        bad_vocabularies.add(s_tag)
                    else:
                        vocabularies.add(s_tag)
                else:
                    continue

vocabularies_list = sorted(list(vocabularies))

# assert all the words are in
for s_word in predefined_vocabularies:
    if s_word not in vocabularies:
        print("{} was not found in dictionary".format(s_word))

idx2word, word2idx = string_list2dict(vocabularies_list)


save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/vocabularies_complete_dict_from_train.pkl', {'idx2word': idx2word, 'word2idx': word2idx})
