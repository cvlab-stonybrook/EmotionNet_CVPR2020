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

data_split = 'train'
tag_label_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_tag_labels_{}_dict.pkl'.format(data_split))
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
# predefined_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/preprocess_selftrained_wordnet_word2vecs_v2.py')
word_embedding_states = torch.load('/home/zwei/Dev/AttributeNet3/TextClassification/models/model_best.pth.tar')
predefined_vocabulary = word_embedding_states['tag2idx']
processed_cids = set()
tag_label_tag_full_list = []

for s_raw_file in tqdm.tqdm(raw_annotation_files):
    raw_annotations = load_json_list(s_raw_file)
    for s_raw_annotation in raw_annotations:
        image_cid = s_raw_annotation['cid']
        if image_cid not in tag_label_dict or image_cid in processed_cids:
            continue
        else:
            processed_cids.add(image_cid)
            tag_label_annotation = tag_label_dict[image_cid]
            raw_tags = remove_hat(s_raw_annotation['tags'])
            updated_tags = []
            for s_tag in raw_tags:
                if s_tag in predefined_vocabulary:
                    updated_tags.append(predefined_vocabulary[s_tag])
            if len(updated_tags) < 1:
                continue

            tag_label_tag_full_list.append([tag_label_annotation[0], tag_label_annotation[1], updated_tags])


print("total training data {}\t after processing {}".format(len(tag_label_dict), len(tag_label_tag_full_list)))
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_tag_label_tag_full_{}.pkl'.format(data_split), tag_label_tag_full_list)
