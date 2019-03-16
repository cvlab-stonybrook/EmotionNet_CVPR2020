# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 23/Feb/2019 11:04

from PyUtils.pickle_utils import loadpickle, save2pickle
from AdobeStockTools.TagUtils import is_good_tag
from PyUtils.dict_utils import string_list2dict
keyword_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl')
pretrained_vocabulary = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/pre_extract_w2v/params/googlenews_extracted_w2v_wordnet_synsets_py3.pl')
keyword2idx = keyword_vocabulary['key2idx']

selected_words = []
for s_keyword in keyword2idx:
    if s_keyword in pretrained_vocabulary:
        if s_keyword == 'xmas':
            continue
        selected_words.append(s_keyword)
    else:
        print("{} is not in google w2v selected vocabulary".format(s_keyword))


selected_words = sorted(selected_words)
idx2key, key2idx = string_list2dict(selected_words)
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/Emotion_vocabulary.pkl', {'key2idx': key2idx, 'idx2key': idx2key})