# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): check the quality of the annotation using only the selected tags
# Email: hzwzijun@gmail.com
# Created: 11/Feb/2019 10:34

from PyUtils.pickle_utils import loadpickle
from EmotionTag.load_csv_annotations import load_verified
from Datasets.WebEmo.constants import labels as webemo_labels
webemo_images_test = loadpickle('/home/zwei/Dev/AttributeNet3/Datasets/WebEmo/data/test_url_tags_emotion25.pkl')
selected_tags, tag_transfer, _ = load_verified('/home/zwei/Dev/AttributeNet3/EmotionTag/emotion_annotations_csv/Verify_20190210.csv')
selected_tags_set = set(selected_tags)

for s_idx, s_image_cid in enumerate(webemo_images_test):
    s_image_information = webemo_images_test[s_image_cid]
    s_selected_tags = []
    for s_tag in s_image_information[1]:
        if s_tag in selected_tags_set:
            s_selected_tags.append(s_tag)
        elif s_tag in tag_transfer:
            s_selected_tags.append(tag_transfer[s_tag])
        else:
            continue
    print("** {}\t{}".format(s_idx, s_image_cid))
    print(s_image_information[0])
    print("labels:{}".format('; '.join(['{};{};{}'.format(webemo_labels[x][2], webemo_labels[x][1], webemo_labels[x][0]) for x in s_image_information[2]])))
    print("selected tags: {}".format(', '.join(s_selected_tags)))
