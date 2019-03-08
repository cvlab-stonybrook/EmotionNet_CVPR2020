# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): https://projector.tensorflow.org/
# Email: hzwzijun@gmail.com
# Created: 30/Oct/2018 11:39


import gensim
from PyUtils.pickle_utils import loadpickle
import numpy as np
import os
from PyUtils.file_utils import get_dir

save_directory = get_dir("KeywordVisualizations")
word_vectors = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/visualizations/Embeddings/FullVocab_BN_transformed_l2_regularization.pkl')

word_info_file = os.path.join(save_directory, 'projector_words_full_voc_transformed_l2_info.dat')
word_vec_file = os.path.join(save_directory, 'projector_words_full_voc_transformed_l2_vec.dat')
word_vector_matrix = []
# word_vectors = []
with open(word_info_file, 'w') as of_:
    of_.write('word')
    for  s_word in word_vectors:
        s_word_vector = word_vectors[s_word]


        word_vector_matrix.append(s_word_vector)
        of_.write('{}\n'.format(s_word))



with open(word_vec_file, 'w') as of_:
    for s_vec in word_vector_matrix:
        assert np.sum(np.abs(s_vec))>0
        of_.write('{}\n'.format('\t'.join(str(x) for x in s_vec[0])))






