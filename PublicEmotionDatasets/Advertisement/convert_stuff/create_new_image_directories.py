# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create an image directory that only contains the images used in rameswar
# Email: hzwzijun@gmail.com
# Created: 14/Mar/2019 16:42

import shutil
import os
from PyUtils.file_utils import get_dir, get_stem
from PyUtils.pickle_utils import save2pickle

src_image_directory = '/home/zwei/datasets/PublicEmotion/tars/Ads/advt_dataset/images'
target_image_directory = get_dir('/home/zwei/datasets/PublicEmotion/Advertisement/images')


image_list_files = ['/home/zwei/datasets/PublicEmotion/tars/Ads/advt_dataset/train_list.txt',
                    '/home/zwei/datasets/PublicEmotion/tars/Ads/advt_dataset/test_list_2403.txt']

create_directories = []
copy_image_list = []
for s_file in image_list_files:
    image_list = []

    with open(s_file, 'r') as of_:
        lines = of_.readlines()
        for s_line in lines:
            s_line_parts = s_line.strip().split(' ')
            s_image_fullpath = s_line_parts[0]
            s_image_category = int(s_line_parts[1])
            s_image_name_path = os.path.join(*s_image_fullpath.split(os.sep)[-2:])
            s_target_full_path = os.path.join(target_image_directory, s_image_name_path)
            image_list.append([s_image_name_path, s_image_category])
            s_src_full_path = os.path.join(src_image_directory, s_image_name_path)
            copy_image_list.append([s_src_full_path, s_target_full_path])
            create_directories.append(os.path.dirname(s_target_full_path))
    save2pickle('{}.pkl'.format(get_stem(s_file)), image_list)
for s_created_directory in list(set(create_directories)):
    if not os.path.exists(s_created_directory):
        os.mkdir(s_created_directory)

import tqdm
import shutil

for s_copy_item in tqdm.tqdm(copy_image_list):
    shutil.copyfile(s_copy_item[0], s_copy_item[1])






