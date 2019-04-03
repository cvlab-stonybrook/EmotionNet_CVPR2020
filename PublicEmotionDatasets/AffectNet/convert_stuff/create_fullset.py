# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 14:46

from PyUtils.pickle_utils import loadpickle
train_set = loadpickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/training.pkl')
val_set = loadpickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/validation.pkl')