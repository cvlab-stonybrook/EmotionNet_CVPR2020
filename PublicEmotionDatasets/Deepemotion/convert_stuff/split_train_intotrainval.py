# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Mar/2019 11:15

from PyUtils.pickle_utils import loadpickle, save2pickle
import random
random.seed(0)

train_data = loadpickle('/home/zwei/datasets/PublicEmotion/Deepemotion/z_data/train_8.pkl')

split = len(train_data)//10
random.shuffle(train_data)
train_val_data = train_data[:split]
train_train_data = train_data[split:]

save2pickle('/home/zwei/datasets/PublicEmotion/Deepemotion/z_data/train_8_90_list.pkl', train_train_data)
save2pickle('/home/zwei/datasets/PublicEmotion/Deepemotion/z_data/train_8_10_list.pkl', train_val_data)
print("DB")