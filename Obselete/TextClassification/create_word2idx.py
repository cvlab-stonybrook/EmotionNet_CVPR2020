# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Feb/2019 08:53


import torch
import gensim
import torch.nn as nn
from PyUtils.pickle_utils import loadpickle

model_path = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/word2vec.model'

model = gensim.models.word2vec.Word2Vec.load(model_path)
digit_word2idx = {token: token_index for token_index, token in enumerate(model.wv.index2word)}
digit2word = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/vocabularies_complete_dict_from_train.pkl')['word2idx']

weights = torch.FloatTensor(model.vectors)
embedding = nn.Embedding.from_pretrained(weights)
print("DB")