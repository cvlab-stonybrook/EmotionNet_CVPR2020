# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Feb/2019 20:50
from PyUtils.dict_utils import string_list2dict

emotion_categories = sorted(['fear', 'sadness', 'excitement', 'amusement', 'anger', 'awe', 'contentment', 'disgust'])
idx2emotion, emotion2idx = string_list2dict(emotion_categories)