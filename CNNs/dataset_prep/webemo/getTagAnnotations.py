# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 21/Oct/2018 19:53

from PyUtils.json_utils import load_json_list
from PyUtils.pickle_utils import loadpickle, save2pickle
import glob
import os
from AdobestockTools.spellchecker import rawtags_cleanup
import tqdm
split = 'train'
CIDs = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/{}-CIDs.pkl'.format(split))
CIDs = set(CIDs)
nrc_annotation_files = glob.glob(os.path.join('/home/zwei/datasets/adobestock_nrc/raw', '*.json'))
random_annotation_files = glob.glob(os.path.join('/home/zwei/datasets/adobestock_random/raw', '*.json'))
full_annotation_files = nrc_annotation_files + random_annotation_files

CIDAnnotaitons = {}
for s_annotation_file in tqdm.tqdm(full_annotation_files):
    s_annotations = load_json_list(s_annotation_file)
    for s_annotation in s_annotations:
        sCID = s_annotation['cid']
        if sCID in CIDs:
            raw_tags = rawtags_cleanup(s_annotation['tags'])
            raw_caption = s_annotation['title']
            CIDAnnotaitons[sCID] = (raw_tags, raw_caption)
print("{}: Total {}, Find {} CIDs existing!".format(split, len(CIDs), len(CIDAnnotaitons)))
save2pickle('ExistCIDTags+Captions-{}.pkl'.format(split), CIDAnnotaitons)

        # print("DEB")
#train: Total 183372, Find 178707 CIDs existing!
#test:  Total 51212, Find 50039 CIDs existing!