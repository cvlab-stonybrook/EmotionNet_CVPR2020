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


def get_w2v_embeddings_from_pretrained_googlenews_wordnet(pretrained_embedding_fpath, save_path):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    tic = time.time()
    print('Please wait ... (it could take a while to load the file : {})'.format(pretrained_embedding_fpath))
    # model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embedding_fpath, binary=True)
    model = gensim.models.word2vec.Word2Vec.load(pretrained_embedding_fpath)

    print('Done.  (time used: {:.1f}s)\n'.format(time.time()-tic))

    embedding_weights = {}
    found_cnt = 0

    for word in tqdm.tqdm(model.wv.vocab, desc="Filtering Words Using WordNet"):
        # if has_digits(word) or len(word)<3:
        #     continue
        # if len(wordnet.synsets(word))>0:
            embedding_weights[word] = model.wv.word_vec(word)
            found_cnt += 1


    save2pickle(save_path, embedding_weights)





def main():

    default_public_w2v_file = '/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_models/word2vec_shuffle.model'

    directory = "./params"
    fpath_pretrained_extracted = os.path.expanduser("{}/selftrained_shuffle_extracted_w2v_wordnet_synsets_py{}.pl".format(directory, sys.version_info.major))

    get_w2v_embeddings_from_pretrained_googlenews_wordnet(pretrained_embedding_fpath=default_public_w2v_file, save_path=fpath_pretrained_extracted)

if __name__ == "__main__":
    main()
