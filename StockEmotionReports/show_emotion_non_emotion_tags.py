# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 13:23


from PyUtils.pickle_utils import loadpickle, save2pickle
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
import os

emotion_dicts = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/Emotion_vocabulary.pkl')
emotion2idx = emotion_dicts['key2idx']
idx2emotion = emotion_dicts['idx2key']

image_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')

specific_catgory = "stressed"
image_cid2url = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/other/dataset_image_urls.pkl')

tag_dicts = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/vocab_dict.pkl')
idx2tag = tag_dicts['idx2tag']
import tqdm
print("{}".format(specific_catgory))

for s_data_idx, s_data in enumerate(image_data):
    if s_data_idx %1000 != 0:
        continue
    s_name = s_data[0]
    s_image_cid = int(get_image_cid_from_url(s_name, location=1))
    download_url = image_cid2url[s_image_cid]
    s_name_parts = s_name.split(os.sep)
    if len(s_name_parts) == 2:
        s_tag_ids  = s_data[2]
        full_tags = []
        emotion_tags = []
        for s_tag_id in s_tag_ids:
            tag_name = idx2tag[s_tag_id]
            full_tags.append(tag_name)
            if tag_name in emotion2idx:
                emotion_tags.append(tag_name)

        print(download_url)
        print(', '.join(full_tags))
        print(', '.join(emotion_tags))
