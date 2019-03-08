# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 25/Feb/2019 16:46

from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
from PyUtils.pickle_utils import loadpickle, save2pickle

selected_list = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_1000x10x5_v3/selected_list_10000.pkl')
selected_10k_cids = set()

for s_item in selected_list:
    s_cid = int(get_image_cid_from_url(s_item[0]))
    selected_10k_cids.add(s_cid)

selected_10k_cids = list(selected_10k_cids)
save2pickle('selected_cids_10000.pkl', selected_10k_cids)