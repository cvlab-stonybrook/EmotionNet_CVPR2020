# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Feb/2019 12:19
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import tqdm
from PyUtils.file_utils import get_dir
from PyUtils.pickle_utils import loadpickle
from gensim import models
import random
random.seed(0)

user_root = os.path.expanduser('~')

training_data = loadpickle(os.path.join(user_root, 'Dev/AttributeNet3/LanguageData/raw_tag_sentences.pkl'))
save_directory = get_dir( os.path.join(user_root, 'Dev/AttributeNet3/LanguageData/word2vec_models'))

training_sentences = training_data['data']
max_len = training_data['max_len']

embedding_dim = 300
window_size = min(max_len, 50)

texts = []
shuffle_times = 5




def random_drop(s_text, drop_rate=0.1):
    updated_text = []
    for s_tag in s_text:
        p = random.uniform(0, 1)
        if p >= drop_rate:
            updated_text.append(s_tag)
    return updated_text


for s_cid in tqdm.tqdm(training_sentences, total=len(training_sentences), desc="Creating texts for trianing"):
    s_text = training_sentences[s_cid]
    aug_times = 0
    texts.append(s_text)
    while aug_times < shuffle_times:
        random.shuffle(s_text)
        random_drop_text = random_drop(s_text)
        texts.append(random_drop_text)
        aug_times += 1

print("Training on {} Sentences, shuffle rate {}".format(len(texts), shuffle_times))
model = models.Word2Vec(texts, size=embedding_dim, window=window_size, workers=128, sg=1, hs=0, negative=15, iter=10, alpha=0.025, min_count=100)
model.save(os.path.join(save_directory, "word2vec_raw_shuffle_v2.model"))
print("Done")
