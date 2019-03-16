# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 14/Mar/2019 17:20

import tqdm
import os
from PyUtils.pickle_utils import loadpickle
from PublicEmotionDatasets.Advertisement.constants import idx2emotion
from PyUtils.file_utils import get_stem
import shutil

image_directory = '/home/zwei/datasets/PublicEmotion/Advertisement/images'

image_annotations = loadpickle('/home/zwei/datasets/PublicEmotion/Advertisement/train_list.pkl')
target_directories = '/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/Advertisement/convert_stuff/examples'

for idx, s_image_annotation in enumerate(tqdm.tqdm(image_annotations)):
    if idx % 500 == 0:
        s_image_path = os.path.join(image_directory, s_image_annotation[0])
        if os.path.exists(s_image_path):
            extension_name = s_image_annotation[0].strip().split('.')[1]
            target_name = '{}-{}-{}.{}'.format(os.path.dirname(s_image_annotation[0]), get_stem(s_image_annotation[0]), idx2emotion[s_image_annotation[1] + 1], extension_name)
            target_path = os.path.join(target_directories, target_name)
            shutil.copyfile(s_image_path, target_path)
            print("{}\t{}\t{}".format(s_image_annotation[0], s_image_annotation[1], idx2emotion[s_image_annotation[1] + 1]))
        else:
            print("!!!! {} Not Found!".format(s_image_path))