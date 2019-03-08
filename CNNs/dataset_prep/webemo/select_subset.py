# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 23/Oct/2018 17:56
import tqdm
import os

import PyUtils.file_utils as file_utils
from PyUtils.pickle_utils import loadpickle, save2pickle
image_directory = '/home/zwei/datasets/emotion_datasets/webemo/webemo-images-256'
data = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/test-CIDs-fullinfo.pkl')
from CNNs.dataset_prep.webemo.constants import emotion_type_25
selected_categories = {}
from PIL import Image
import shutil

target_image_directory = file_utils.get_dir('/home/zwei/datasets/emotion_datasets/webemo/selected_annotated')
for item in tqdm.tqdm(data[:100]):
    selected_categories[item[0]] = item
    src_image = os.path.join(image_directory, item[-1])
    target_image_name = item[-1].replace('/', '-')
    if item[5] == 1:
        target_image_name = 'p-'+target_image_name
    else:
        target_image_name = 'n-'+target_image_name

    dst_image = os.path.join(target_image_directory, target_image_name)

    shutil.copyfile(src_image, dst_image)

save_file_name = os.path.join(target_image_directory, 'PandaInfo.pkl')
save2pickle(save_file_name, selected_categories)
print("DEB")

