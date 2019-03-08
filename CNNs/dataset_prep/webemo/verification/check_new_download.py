# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 22/Nov/2018 17:38

import sys, os
import glob
from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.file_utils import get_stem
webemo_dir = '/home/zwei/Downloads/WEBEmo'
webemo_files = glob.glob(os.path.join(webemo_dir,'*.txt'))

CIDs = {}
split = 'test25'

for file_idx, s_file in enumerate(webemo_files):
    if split in s_file:
        print("Processing:[ {} | {} ] {:s}".format(file_idx, len(webemo_files), s_file))
        total_files = 0
        added_files = 0
        with open(s_file, 'r') as of_:
            for s_line in of_:
                contents = s_line.strip().split(' ')
                cid = get_stem(contents[0])
                # cid = contents[0]

                total_files += 1
                if cid not in CIDs:
                    # print("Add {:s}".format(contents[0]))
                    CIDs[cid] = [contents[1]]
                    added_files += 1
                else:
                    CIDs[cid].append(contents[1])

        print("{:s}: total-files: {}\t Added {}\t".format(s_file, total_files, added_files))
print("Total files for {}: \t{}".format(split, len(CIDs)))
save2pickle('{}-CIDs.pkl'.format(get_stem(split)), (CIDs))
