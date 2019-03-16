# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 15/Mar/2019 15:50

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm

annotation_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_test.pkl')
tagidx_data = []
for s_data in tqdm.tqdm(annotation_data):
    tagidx_data.append([s_data[0], s_data[2]])

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tagidx_36534_test.pkl', tagidx_data)

