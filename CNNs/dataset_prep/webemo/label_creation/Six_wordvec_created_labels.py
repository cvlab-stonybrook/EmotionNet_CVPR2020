# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 31/Oct/2018 18:55

from CNNs.dataset_prep.webemo.constants import emotion_type_6
from PyUtils.dict_utils import string_list2dict
from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
from lda2vec.word2vec_NN.external_utils import NearestN_sim
import gensim
import os
import numpy as np

idx2emotion, emotion2idx = string_list2dict(emotion_type_6)
split = 'test'
topN = 5
data = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/{}-CIDs-fullinfo.pkl'.format(split))

save_directory = '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data'

save_file = os.path.join(save_directory, 'Emotion6-{}-w2v-NN{}.pkl'.format(split, topN))

model_directory = '/home/zwei/Dev/AttributeNet/lda2vec/webemo-train'
w2v_model = gensim.models.word2vec.Word2Vec.load(os.path.join(model_directory, "word2vec.model"))
word_data = loadpickle(os.path.join(model_directory, 'words.pkl'))
word_decoder = word_data['id2word']
word_encoder = word_data['word2id']
word_count = word_data['wordcounts']

save_data = []

for s_data in tqdm.tqdm(data):
    s_path = s_data[-1]
    s_tags = s_data[1]
    s_emotion_scores = np.zeros(len(emotion2idx))
    for s_emotion in emotion2idx:
        s_score = NearestN_sim(s_emotion, s_tags, w2v_model, word_encoder, topN=topN)
        s_emotion_scores[emotion2idx[s_emotion]] = s_score

    max_idx = np.argmax(s_emotion_scores)
    save_data.append((s_path, max_idx))

save2pickle(save_file, save_data)


