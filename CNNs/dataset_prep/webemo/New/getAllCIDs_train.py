# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): get all CIDs listed in Rameswar's file
# Email: hzwzijun@gmail.com
# Created: 15/Oct/2018 19:02


import sys, os
import glob
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.file_utils import get_stem
webemo_dir = '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/New/raw_data'


CIDs = set()
Labels = {}

split = 'train'

webemo_file = os.path.join(webemo_dir,'{}25.txt'.format(split))


raw_sample_count = 0
with open(webemo_file, 'r') as of_:
    for s_line in of_:
        raw_sample_count += 1

        contents = s_line.strip().split(' ')
        cid = int(get_stem(contents[0]).split('_')[2])
        if cid not in CIDs:
            CIDs.add(cid)
            Labels[cid] = [int(contents[1])]
        else:
            Labels[cid].append(int(contents[1]))


print("total raw samples: {}\t Added {}\t".format(raw_sample_count, len(CIDs)))
assert len(CIDs) == len(Labels)

print("Total files for {}: \t{}".format(split, len(CIDs)))
save_data = {'CIDs': CIDs, 'labels': Labels}
save2pickle('data/{}-RawCIDs.pkl'.format(get_stem(split)), save_data)

