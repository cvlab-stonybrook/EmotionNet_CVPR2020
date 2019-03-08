# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 12/Feb/2019 09:37

from PyUtils.json_utils import load_json_list
import glob
from PyUtils.dict_utils import get_value_sorted_dict
import os
import tqdm
from PyUtils.file_utils import get_stem
from PyUtils.pickle_utils import save2pickle

raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))

tag_frequencies = {}

for s_raw_idx, s_raw_file in enumerate(tqdm.tqdm(raw_annotation_files, desc="Processing Files")):

    s_annotations = load_json_list(s_raw_file)
    tag_keyword = get_stem(s_raw_file)


    if len(s_annotations) > 6000:
        tag_image_cids = set()
        for s_annotation in s_annotations:
            s_cid = s_annotation['cid']
            tag_image_cids.add(s_cid)

        if len(s_annotations)!= len(tag_image_cids):
            print("{} has repeated cids in its search list".format(tag_keyword))

        tag_frequencies[tag_keyword] = list(tag_image_cids)

tag_frequencies_count = {}
for s_tag in tag_frequencies:
    tag_frequencies_count[s_tag] = len(tag_frequencies[s_tag])

tag_frequencies_count = get_value_sorted_dict(tag_frequencies_count, reverse=True)

for s_idx, s_tag in enumerate(tag_frequencies_count):
    print("{}\t{}\t{}\t{}".format(s_idx, s_tag, len(tag_frequencies[s_tag]), tag_frequencies_count[s_tag]))

print("Total: {}".format(len(tag_frequencies)))
save2pickle('data_v2/tag_frequencies_selected.pkl', tag_frequencies)