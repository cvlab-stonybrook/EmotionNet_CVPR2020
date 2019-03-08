# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 22/Feb/2019 11:19


from PyUtils.json_utils import load_json_list
from PyUtils.pickle_utils import loadpickle, save2pickle
import glob, os
from nltk.corpus import wordnet
from AdobeStockTools.TagUtils import remove_hat, has_digits
import tqdm
from collections import Counter
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))

preloaded_dict = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/pre_extract_w2v/params/googlenews_extracted_w2v_wordnet_synsets_py3.pl')
preloaded_keys = set(list(preloaded_dict.keys()))
processedCIDs = set()
missed_tags = []
for s_file in tqdm.tqdm(raw_annotation_files):
    s_raw_annotations = load_json_list(s_file)
    for s_raw_annotation in s_raw_annotations:
        s_image_cid = s_raw_annotation['cid']
        if s_image_cid in processedCIDs:
            continue
        else:
            processedCIDs.add(s_image_cid)
            s_tags = remove_hat(s_raw_annotation['tags'])
            for s_tag in s_tags:
                if len(s_tag)<3 or has_digits(s_tag):
                    continue
                if len(wordnet.synsets(s_tag)) >= 1:
                    if s_tag not in preloaded_keys:
                        missed_tags.append(s_tag)


missed_tag_counts = Counter(missed_tags)
save2pickle('Missed_Tag_Counts_Tmp.pkl', missed_tag_counts)
print("DEB")
