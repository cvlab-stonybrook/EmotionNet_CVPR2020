# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 14/Mar/2019 17:10
import tqdm
import os
from PyUtils.pickle_utils import loadpickle
from PublicEmotionDatasets.Advertisement.constants import idx2emotion
image_directory = '/home/zwei/datasets/PublicEmotion/Advertisement/images'

image_annotations = loadpickle('/home/zwei/datasets/PublicEmotion/Advertisement/train_list.pkl')

for s_image_annotation in tqdm.tqdm(image_annotations):
    s_image_path = os.path.join(image_directory, s_image_annotation[0])
    if os.path.exists(s_image_path):
        print("{}\t{}\t{}".format(s_image_annotation[0], s_image_annotation[1], idx2emotion[s_image_annotation[1] + 1]))
    else:
        print("!!!! {} Not Found!".format(s_image_path))
