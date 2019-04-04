# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 15:02

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm

train_files = loadpickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/training_exist.pkl')

categorical_dict = {}

for s_file in tqdm.tqdm(train_files):
    if s_file[1] in categorical_dict:
        categorical_dict[s_file[1]].append(s_file[0])
    else:

        categorical_dict[s_file[1]] =  [s_file[0]]


save2pickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/training_categorical_dicts.pkl', categorical_dict)
