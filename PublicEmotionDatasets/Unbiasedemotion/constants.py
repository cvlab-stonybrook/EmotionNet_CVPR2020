# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 20:52

from PyUtils.dict_utils import string_list2dict

emotion_categories = sorted(['love', 'anger', 'surprise', 'joy', 'sadness', 'fear'])
idx2emotion, emotion2idx = string_list2dict(emotion_categories)