# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Feb/2019 10:46

# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Feb/2019 10:29


from nltk.corpus import stopwords

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.json_utils import load_json_list

import tqdm
from AdobeStockTools.TagUtils import remove_hat
import glob, os


image_cids = set(loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_train.pkl'))
vocabularies_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/vocabularies_complete_dict_from_train.pkl')
word2idx = vocabularies_dict['word2idx']

raw_annotaiton_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))

cid_sentences = {}
for s_raw_annotation_file in tqdm.tqdm(raw_annotaiton_files):
    keyword_annotations = load_json_list(s_raw_annotation_file)
    for s_annotation in keyword_annotations:
        s_cid = s_annotation['cid']
        if s_cid in cid_sentences:
            continue
        if s_cid not in image_cids:
            continue

        s_sentence = []
        s_tags = remove_hat(s_annotation['tags'])
        for s_tag in s_tags:
            if s_tag in word2idx:
                s_sentence.append(word2idx[s_tag])
        if len(s_sentence) > 5:
            cid_sentences[s_cid] = s_sentence

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/sentences_train.pkl', cid_sentences)

