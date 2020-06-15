# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Feb/2019 17:12


from PyUtils.pickle_utils import loadpickle
import gensim
import os
import numpy as np
from PyUtils.dict_utils import string_list2dict
from sklearn.metrics.pairwise import cosine_similarity

anchor_annotations  = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/etag2idx.pkl')
anchor_data = anchor_annotations['key2idx']
anchor_words = list(anchor_data.keys())
anchor_word_set = set(anchor_words)
# model_path = '/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/params/googlenews_extracted_w2v_wordnet_synsets_py3.pl'
model_path = '/home/zwei/Dev/AttributeNet3/TextClassificationV2/ckpts/selftrained.pkl'


word2vec_model = loadpickle(model_path)


word2vec_matrix = []
word_list = []
for s_word in word2vec_model:
    word_list.append(s_word)
    word2vec_matrix.append(word2vec_model[s_word])


idx2word, word2idx = string_list2dict(word_list)
word2vec_matrix = np.array(word2vec_matrix)


for word_idx, s_word in enumerate(anchor_words):
    # if s_word not in word2idx:
    #     print("Cannot find {}".format(s_word))
    #     continue
    # s_word_id = str(word2idx[s_word])
    if s_word in word2idx:
        s_word_vec = word2vec_matrix[word2idx[s_word]]
    else:
        print("{} Not found in the learned dict".format(s_word))
        continue

    knns = cosine_similarity(s_word_vec.reshape(1, -1), word2vec_matrix)[0]
    topK = np.argsort(knns)[::-1][:20]
    anchor_nns = []
    for s_knn in topK:
        # if idx2word[s_knn] in anchor_word_set:
            anchor_nns.append([idx2word[s_knn], knns[s_knn] ])
    # print("Anchor: {}".format(s_word))
    # for idx, s_nn_id in enumerate(anchor_nns[:10]):
    #     print(" ** {}\t{} ({:.4f})".format(idx, s_nn_id[0], s_nn_id[1]))
    # TODO: latex print:

    nearest_print = []
    for idx, s_nn_id in enumerate(anchor_nns[1:10]):
        nearest_print.append("\t{} ({:.3f})".format(s_nn_id[0], s_nn_id[1]))
    # print("\item  \\textbf{{ {} }} : {}".format(s_word, ', '.join(nearest_print)))
    print("{}  \\textbf{{ {} }} : {}".format(word_idx, s_word, ', '.join(nearest_print)))

print("DEB")