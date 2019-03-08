# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): check if there is overlap between training and testing
# Email: hzwzijun@gmail.com
# Created: 22/Nov/2018 17:15

from PyUtils.pickle_utils import loadpickle
import os


train_file = '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/verification/train25-CIDs.pkl'
train_CIDs = loadpickle(train_file)

test_file = '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/verification/test25-CIDs.pkl'
test_CIDS = loadpickle(test_file)

# test_CID_set = set(test_CIDS)
# assert len(test_CID_set) == len(test_CIDS), "DOUBLE CHECK"
count = 0
for s_CID in train_CIDs:
    if s_CID in test_CIDS:
        count += 1
        print("{} Both in train and val! train labels: {}, val labels: {}".format(s_CID, ', '.join(train_CIDs[s_CID]), ', '.join(test_CIDS[s_CID])))

print("Train: {}\tTest: {}\tOverlap: {}".format(len(train_CIDs), len(test_CIDS), count))

print("DB")