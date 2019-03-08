# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Feb/2019 17:12


from PyUtils.pickle_utils import loadpickle
import gensim
import os


anchor_annotations  = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/E_vocabulary.pkl')
anchor_data = anchor_annotations['key2idx']
anchor_words = list(anchor_data.keys())
anchor_word_set = set(anchor_words)
model_path = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/word2vec/word2vec.model'


word2vec_model = gensim.models.word2vec.Word2Vec.load(model_path)
# word2vec_aux = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/vocabularies_complete_dict_from_train.pkl')
# idx2word = word2vec_aux['idx2word']
# word2idx = word2vec_aux['word2idx']


for s_word in anchor_words:
    # if s_word not in word2idx:
    #     print("Cannot find {}".format(s_word))
    #     continue
    # s_word_id = str(word2idx[s_word])
    knns = word2vec_model.most_similar(positive=s_word, topn=20)
    anchor_nns = []
    for s_knn in knns:
        if s_knn[0] in anchor_word_set:
            anchor_nns.append([s_knn[0], s_knn[1]])
    # print("Anchor: {}".format(s_word))
    # for idx, s_nn_id in enumerate(anchor_nns[:10]):
    #     print(" ** {}\t{} ({:.4f})".format(idx, s_nn_id[0], s_nn_id[1]))
    # TODO: latex print:

    nearest_print = []
    for idx, s_nn_id in enumerate(anchor_nns[:10]):
        nearest_print.append("\t{} ({:.3f})".format( s_nn_id[0], s_nn_id[1]))
    print("\item  \\textbf{{ {} }} : {}".format(s_word, ', '.join(nearest_print)))

print("DEB")