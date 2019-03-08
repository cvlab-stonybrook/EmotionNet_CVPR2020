# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 15:57
import os
from PyUtils.pickle_utils import loadpickle

user_root = os.path.expanduser('~')
dataset_dir = os.path.join(user_root, 'datasets/PublicEmotion', 'Deepemotion')

emotion_categories = sorted(['fear', 'sadness', 'excitement', 'amusement', 'anger', 'awe', 'contentment', 'disgust'])
train = loadpickle(os.path.join(dataset_dir, 'train.pkl'))
train_sample = loadpickle(os.path.join(dataset_dir, 'train_sample.pkl'))
test = loadpickle(os.path.join(dataset_dir, 'test.pkl'))
print("Done")