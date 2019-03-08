# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 24/Feb/2019 18:21

# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 24/Feb/2019 17:25

from PyUtils.json_utils import load_json_list
from PyUtils.pickle_utils import loadpickle, save2pickle
import glob, os, tqdm
from AdobeStockTools.TagUtils import remove_hat, keepGoodTags
from PyUtils.file_utils import get_stem
from PyUtils.dict_utils import get_key_sorted_dict, get_value_sorted_dict
import random
random.seed(0)

candidate_cids = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_1000x10x5_v3/data/non_repeat_candidate_cids.pkl')
already_annotated_cids = loadpickle('selected_cids_10000.pkl')
already_annotated_cids_set = set(already_annotated_cids)
assert len(already_annotated_cids) == len(already_annotated_cids_set)
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))




annotation_data = {}
for s_annotation_file in tqdm.tqdm(raw_annotation_files):
    s_keyword = get_stem(s_annotation_file)
    s_raw_annotations = load_json_list(s_annotation_file)
    for s_raw_annotation in s_raw_annotations:
        s_image_cid = s_raw_annotation['cid']
        if s_image_cid not in candidate_cids or s_image_cid in annotation_data:
            continue
        else:
            if s_image_cid in already_annotated_cids:
                continue
            s_tags = keepGoodTags(remove_hat(s_raw_annotation['tags']))
            s_url = s_raw_annotation['url']
            annotation_data[s_image_cid] = [s_url, s_tags, candidate_cids[s_image_cid][-1]]

annotation_data_list = [annotation_data[s_image_cid] for s_image_cid in annotation_data]
random.shuffle(annotation_data_list)
save2pickle('selected_list_rest_20000.pkl', annotation_data_list)



# new_cids = set()
# for s_string in annotations_keystrings:
#     if len(annotations_keystrings[s_string]) > 1:
#         continue
#     else:
#         new_cids.add(annotations_keystrings[s_string][0])
#
# filtered_cids = {}
# label_counts = {}
# for s_cid in candidate_cids:
#     if s_cid not in new_cids:
#         continue
#     else:
#         filtered_cids[s_cid] = candidate_cids[s_cid]
#         for s_label in filtered_cids[s_cid][1]:
#             if s_label in label_counts:
#                 label_counts[s_label] += 1
#             else:
#                 label_counts[s_label] = 1
#
# label_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/E_vocabulary.pkl')
# idx2key = label_vocabulary['idx2key']
# label_counts = get_key_sorted_dict(label_counts)
# for s_label in label_counts:
#     print('{}\t{}'.format(s_label, label_counts[s_label]))
#
# save2pickle('non_repeat_candidate_cids.pkl', filtered_cids)
# print("DB")
