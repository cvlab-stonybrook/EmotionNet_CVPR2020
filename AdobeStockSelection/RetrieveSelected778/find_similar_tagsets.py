# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 13/Feb/2019 10:43

from PyUtils.json_utils import load_json_list
from PyUtils.pickle_utils import loadpickle, save2pickle
import os
from AdobeStockTools.TagUtils import remove_hat
import tqdm

image_meta_directory = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2'

selected_tag_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')
selected_cids = set()
for s_keyword in selected_tag_dict:
    for s_cid in selected_tag_dict[s_keyword]:
        selected_cids.add(s_cid)

processedCIDs = set()
CID_strings_dict = {}
for s_tag in tqdm.tqdm(selected_tag_dict, 'Process Files'):
    image_meta_file = os.path.join(image_meta_directory, '{}.json'.format(s_tag))
    s_tag_meta = load_json_list(image_meta_file)
    for s_image_meta in s_tag_meta:
        s_cid = s_image_meta['cid']
        if s_cid in processedCIDs:
            continue
        else:
            processedCIDs.add(s_cid)
            s_tags = remove_hat(s_image_meta['tags'])
            s_tags_string = ''.join(s_tags)
            if s_tags_string not in CID_strings_dict:
                CID_strings_dict[s_tags_string] = [s_cid]
            else:
                CID_strings_dict[s_tags_string].append(s_cid)

banned_cids = []
for s_string in CID_strings_dict:
    if len(CID_strings_dict[s_string])>1:
        banned_cids.extend(CID_strings_dict[s_string][1:])


save2pickle('data_v2/cids_banned_for_similar_tags.pkl', banned_cids)

