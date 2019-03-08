# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 21/Oct/2018 20:33

from PyUtils.json_utils import load_json_list
from PyUtils.pickle_utils import loadpickle, save2pickle
import glob
import os
from AdobestockTools.spellchecker import rawtags_cleanup
import tqdm
from PyUtils.file_utils import get_stem
split = 'train'
CIDAnnotations = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/ExistCIDTags+Captions-{}.pkl'.format(split))
labelFile = '/home/zwei/datasets/emotion_datasets/webemo/correct_index_files/{}-images-25-6-2-category.txt'.format(split)
CIDLabels = {}
newAnnotations = []
CID_set = set()
with open(labelFile, 'r') as of_:
    for s_line in of_:
        s_contents = s_line.strip().split(' ')
        s_CID = int(get_stem(s_contents[0]))
        s_partial = os.path.join(*(s_contents[0].split(os.sep)[-3:]))
        s_CID_c25 = int(s_contents[1])
        s_CID_c6 = int(s_contents[2])
        s_CID_c2 = int(s_contents[3])
        if s_CID in CIDAnnotations:
            CID_set.add(s_CID)
            tags = CIDAnnotations[s_CID][0]
            title = CIDAnnotations[s_CID][1]
            newAnnotations.append((s_CID, tags, title, s_CID_c25, s_CID_c6, s_CID_c2, s_partial))
print("Got {} {} Data from {} CIDs".format(len(newAnnotations), split, len(CID_set)))
save2pickle('{}-CIDs-fullinfo.pkl'.format(split), newAnnotations)

