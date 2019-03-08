# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Feb/2019 12:19

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import tqdm

from PyUtils.pickle_utils import loadpickle
import os
from gensim import models

training_sentences = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/train_cids_tags.pkl')
save_directory = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/word2vec'
embedding_dim = 300
window_size = 30
texts = []
for s_cid in tqdm.tqdm(training_sentences, total=len(training_sentences), desc="Creating texts for trianing"):
    s_text = [str(j) for j in training_sentences[s_cid]]
    texts.append(s_text)

model = models.Word2Vec(texts, size=embedding_dim, window=window_size, workers=4, sg=1, hs=0, negative=15, iter=10, alpha=0.025)
model.save(os.path.join(save_directory, "word2vec.model"))
print("Done")
