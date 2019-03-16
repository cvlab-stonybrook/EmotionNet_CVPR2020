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
previous_val_cids = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_val_webemo_annotation_excluded_dict.pkl')
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))




processed_cids = set()
annotations_keystrings = {}
for s_annotation_file in tqdm.tqdm(raw_annotation_files):
    s_keyword = get_stem(s_annotation_file)
    s_raw_annotations = load_json_list(s_annotation_file)
    for s_raw_annotation in s_raw_annotations:
        s_image_cid = s_raw_annotation['cid']
        if s_image_cid not in previous_val_cids or s_image_cid in processed_cids:
            continue
        else:
            processed_cids.add(s_image_cid)
            s_tags = keepGoodTags(remove_hat(s_raw_annotation['tags']))
            s_tags = s_tags[0: min(10, len(s_tags))]
            s_tags_string = '-'.join(s_tags)
            if s_tags_string in annotations_keystrings:
                annotations_keystrings[s_tags_string].append(s_image_cid)
            else:
                annotations_keystrings[s_tags_string] = [s_image_cid]
            # if s_image_cid in new_candidate_cids:
            #     new_candidate_cids[s_image_cid][-1].append(s_keyword)
            # else:

non_repeat_cids = set()
for s_string in annotations_keystrings:
    if len(annotations_keystrings[s_string]) > 1:
        continue
    else:
        non_repeat_cids.add(annotations_keystrings[s_string][0])

filtered_cids = {}
label_counts = {}
for s_cid in previous_val_cids:
    if s_cid not in non_repeat_cids:
        continue
    else:
        filtered_cids[s_cid] = previous_val_cids[s_cid]
        for s_label in filtered_cids[s_cid][1]:
            if s_label in label_counts:
                label_counts[s_label] += 1
            else:
                label_counts[s_label] = 1

label_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/Emotion_vocabulary.pkl')
idx2key = label_vocabulary['idx2key']
label_counts = get_key_sorted_dict(label_counts)
for s_label in label_counts:
    print('{}\t{}'.format(s_label, label_counts[s_label]))

print("total cids: {}".format(len(filtered_cids)))
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_val_updated_dict.pkl', filtered_cids)
print("DB")







