# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 02/Nov/2018 17:32
import os
from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm

data_split = 'test'

full_annotation_data = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/{}-CIDs-fullinfo.pkl'.format(data_split))
feature_data = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/feature-{}.pkl'.format(data_split))

save_data = {}
for s_data in tqdm.tqdm(full_annotation_data, desc="Preparing {} for WebEmo ImageWord2Vec Train".format(data_split)):
    sCID = s_data[0]
    s_rel_path = s_data[-1]
    s_tags = s_data[1]
    s_title = s_data[2]
    if s_rel_path in feature_data:
        s_feature = feature_data[s_rel_path]
        s_tosave = {}
        s_tosave['image_feature'] = s_feature
        s_tosave['tags'] = s_tags
        s_tosave['title'] = s_title
        s_tosave['rel_path'] = s_rel_path
        save_data[sCID] = s_tosave


save2pickle(os.path.join('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/{}-imageword2vec.pkl'.format(data_split)), save_data)

print("Done")


