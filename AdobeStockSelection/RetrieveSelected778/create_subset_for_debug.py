# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create a subset with only 1 classes for debug
# Email: hzwzijun@gmail.com
# Created: 17/Feb/2019 19:25

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import string_list2dict
import tqdm

data_split = 'val'
classes_name = ['depression', 'happy', 'angry', 'surprise', 'disappointed', 'contemplate', 'joy', 'crazy',
                'jealous', 'relaxed']
classes_name = sorted(classes_name)
idx2keyword, keyword2idx = string_list2dict(classes_name)

fullidx2keyword = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag-idx-conversion.pkl')['idx2key']
dataset_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/CNNsplit_{}.pkl'.format(data_split))

subset_list = []
for s_data in tqdm.tqdm(dataset_list):
    s_rel_path = s_data[0]
    s_tag_ids = s_data[1]


    updated_tag_ids = []
    for s_tag_id in s_tag_ids:
        if fullidx2keyword[s_tag_id] not in keyword2idx:
            continue
        updated_tag_ids.append( keyword2idx[fullidx2keyword[s_tag_id]])
    if len(updated_tag_ids) < 1:
        continue

    subset_list.append([s_rel_path, updated_tag_ids])

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/CNNsplit_{}.pkl'.format(data_split), subset_list)
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/tag-idx-conversion.pkl', {'idx2key': idx2keyword,
                                                                                                                                 'key2idx':keyword2idx})

