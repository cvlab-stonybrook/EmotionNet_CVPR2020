# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Mar/2019 12:10

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm, os
from PyUtils.file_utils import get_stem
previous_data_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_tag_label_tag_full_train.pkl')
previous_data_dict = {}
current_data_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v7_690_face/CNNsplit_tag_labels_train+face.pkl')
for s_data in tqdm.tqdm(previous_data_list, desc="Creating Dict"):
    x_cid = int(get_stem(s_data[0]).split('_')[1])
    previous_data_dict[x_cid] = s_data


updated_data = []
for s_data in tqdm.tqdm(current_data_list, desc="Creating new data"):
    x_cid = int(get_stem(s_data[0]).split('_')[1])
    if x_cid in previous_data_dict:
        updated_data.append([s_data[0], previous_data_dict[x_cid][1], previous_data_dict[x_cid][2]])



save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v7_690_face/CNNsplit_tag_labels_train+face_full_tag.pkl', updated_data)