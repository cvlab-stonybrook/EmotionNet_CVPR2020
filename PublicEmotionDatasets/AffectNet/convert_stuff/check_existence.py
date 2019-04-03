# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 14:47

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.file_utils import get_dir
from AdobeStockTools.CopyUtils import mThreadCopy

import os
files = loadpickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/validation.pkl')
target_dir = '/home/zwei/datasets/tarPublicEmotion/AffectNet/images-256'
src_dir = '/home/zwei/datasets/tarPublicEmotion/AffectNet/Manually_Annotated_Images_256'

selected_files = []
selected_copies = []
subdirs = set()
for s_file in files:
    src_path = os.path.join(src_dir, s_file[0])
    if os.path.exists(src_path):
        dst_path = os.path.join(target_dir, s_file[0])
        selected_files.append(s_file)
        subdirs.add(os.path.dirname(dst_path))
        selected_copies.append([src_path, dst_path])



for s_dir in subdirs:
    get_dir(s_dir)

import tqdm
from multiprocessing import Pool
num_jobs = 100

pool = Pool(processes=num_jobs)

for s_status in tqdm.tqdm(pool.imap_unordered(mThreadCopy, selected_copies), total=len(selected_copies)):
    if not s_status[0]:
        print(s_status[1])

save2pickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/validation_exist.pkl', selected_files)
