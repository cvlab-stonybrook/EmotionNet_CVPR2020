# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 08/Mar/2019 09:19

from PyUtils.dict_utils import string_list2dict


emotion_list = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
                'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
                'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity',
                'Suffering', 'Surprise', 'Sympathy', 'Yearning']

idx2emorion, emorion2idx = string_list2dict(emotion_list)





