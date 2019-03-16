# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): convert gensim models to pkl file saving the dict
# Email: hzwzijun@gmail.com
# Created: 13/Mar/2019 16:51

import os
import sys
import time
import numpy as np
import pickle
import gensim
from nltk.corpus import wordnet
from PyUtils.pickle_utils import save2pickle
import tqdm
import logging
from nltk.corpus import wordnet
from AdobeStockTools.TagUtils import has_digits
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_embedding_dict_from_gensim(gensim_fpath, filter_function=None):



    tic = time.time()
    print('Please wait ... (it could take a while to load the file : {})'.format(gensim_fpath))
    model = gensim.models.word2vec.Word2Vec.load(gensim_fpath)

    print('Done.  (time used: {:.1f}s)\n'.format(time.time()-tic))

    dict_word_embedding = {}

    found_cnt = 0
    for word in tqdm.tqdm(model.wv.vocab, desc="Extracting words"):
            if filter_function is None:
                dict_word_embedding[word] = model.wv.word_vec(word)
                found_cnt += 1
            else:
                if filter_function(word):
                    dict_word_embedding[word] = model.wv.word_vec(word)
                    found_cnt += 1

    print("{} words extracted".format(found_cnt))

    return dict_word_embedding






def main():

    gensim_model = '/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_models/word2vec_shuffle.model'
    directory = "/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_dicts"
    fpath_pretrained_extracted = os.path.join(directory, "selftrained_shuffle_w2v_dict.pl")

    embedding_dict = get_embedding_dict_from_gensim(gensim_fpath=gensim_model)
    save2pickle(fpath_pretrained_extracted, embedding_dict)


if __name__ == "__main__":
    main()
