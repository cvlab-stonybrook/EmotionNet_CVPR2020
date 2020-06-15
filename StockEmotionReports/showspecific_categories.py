# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 18/Mar/2019 21:22

from PyUtils.pickle_utils import loadpickle, save2pickle
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
import os

emotion_dicts = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/etag2idx.pkl')
emotion2idx = emotion_dicts['key2idx']
idx2emotion = emotion_dicts['idx2key']

image_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')

specific_catgory = "stressed"
image_cid2url = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/other/dataset_image_urls.pkl')

import tqdm
print("{}".format(specific_catgory))
for s_data in tqdm.tqdm(image_data):
    s_name = s_data[0]
    s_image_cid = int(get_image_cid_from_url(s_name, location=1))
    download_url = image_cid2url[s_image_cid]
    s_name_parts = s_name.split(os.sep)
    if len(s_name_parts) == 2:
        for s_emotion_idx in s_data[1]:
            if idx2emotion[s_emotion_idx] == specific_catgory:
                print(download_url)




