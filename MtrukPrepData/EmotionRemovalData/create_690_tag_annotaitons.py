# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 25/Feb/2019 16:49

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob, os, tqdm
from PyUtils.json_utils import load_json_list
from AdobeStockTools.TagUtils import remove_hat
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
from PyUtils.dict_utils import get_value_sorted_dict
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
selected_list = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_1000x10x5_v3/selected_list_10000.pkl')
selected_dict = {}
for s_item in selected_list:
    s_cid = int(get_image_cid_from_url(s_item[0]))
    selected_dict[s_cid] = s_item


processed_cids = set()
created_annotations = []
keyword_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/E_vocabulary.pkl')
key2idx = keyword_vocabulary['key2idx']

tag_counts = {}
for s_raw_annotation_file in tqdm.tqdm(raw_annotation_files):
    s_raw_annotations = load_json_list(s_raw_annotation_file)
    for s_raw_annotation in s_raw_annotations:
        s_cid = s_raw_annotation['cid']
        if s_cid not in selected_dict or s_cid in processed_cids:
            continue
        else:
            processed_cids.add(s_cid)
            s_tags = remove_hat(s_raw_annotation['tags'])
            updated_tags = []
            for s_tag in s_tags:
                if s_tag in key2idx:
                    if s_tag in tag_counts:
                        tag_counts[s_tag] += 1
                    else:
                        tag_counts[s_tag] = 1

                    updated_tags.append(s_tag)
            if len(updated_tags) > 0:
                created_annotations.append([selected_dict[s_cid][0], updated_tags])

            else:
                print("{} has no tags in 690 categories".format(s_cid))


tag_counts = get_value_sorted_dict(tag_counts, reverse=False)
save2pickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionRemovalData/selected_10k_tag_annotations_list.pkl', created_annotations)
print("DONE")


