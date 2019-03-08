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
webemo_dir = '/home/zwei/datasets/emotion_datasets/webemo/'
webemo_files = glob.glob(os.path.join(webemo_dir,'*.txt'))

CIDs = set()
split = 'train'

for file_idx, s_file in enumerate(webemo_files):
    if split in s_file:
        print("Processing:[ {} | {} ] {:s}".format(file_idx, len(webemo_files), s_file))
        total_files = 0
        added_files = 0
        with open(s_file, 'r') as of_:
            for s_line in of_:
                contents = s_line.strip().split(' ')
                cid = int(get_stem(contents[0]))
                total_files += 1
                if cid not in CIDs:
                    # print("Add {:s}".format(contents[0]))
                    CIDs.add(cid)
                    added_files += 1
        print("{:s}: total-files: {}\t Added {}\t".format(s_file, total_files, added_files))
print("Total files for {}: \t{}".format(split, len(CIDs)))
# save2pickle('{}-CIDs.pkl'.format(get_stem(split)), list(CIDs))

