# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 26/Mar/2019 11:42

from PyUtils.pickle_utils import loadpickle

keydicts = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/etag2idx.pkl')

key2idx = keydicts['key2idx']


keys = []
for idx, s_key in enumerate(key2idx):
    keys.append('{}: {}'.format(idx+1, s_key))


print(", ".join(keys))

