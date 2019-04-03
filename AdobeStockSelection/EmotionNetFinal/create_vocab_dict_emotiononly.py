# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Mar/2019 22:05

from PyUtils.pickle_utils import loadpickle, save2pickle

emotion_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/Emotion_vocabulary.pkl')['key2idx']
vocab_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/vocab_dict.pkl')

vocab_tag2idx = vocab_dict['tag2idx']
vocab_idx2tag = vocab_dict['idx2tag']

emotion_tag2idx = {}
emotion_idx2tag = {}

for s_tag in vocab_tag2idx:
    if s_tag in emotion_dict:
        emotion_tag2idx[s_tag] = vocab_tag2idx[s_tag]

for s_idx in vocab_idx2tag:
    if vocab_idx2tag[s_idx] in emotion_dict:
        emotion_idx2tag[s_idx] = vocab_idx2tag[s_idx]

assert len(emotion_idx2tag) == len(emotion_tag2idx) == len(emotion_dict)

save2pickle('emotion-tagidx.pkl', {'tag2idx': emotion_tag2idx, 'idx2tag':emotion_idx2tag})