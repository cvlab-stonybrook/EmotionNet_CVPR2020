# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 23/Feb/2019 10:52


from PyUtils.pickle_utils import loadpickle
from AdobeStockTools.TagUtils import is_good_tag
keyword_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl')
pretrained_vocabulary = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/pre_extract_w2v/params/googlenews_extracted_w2v_wordnet_synsets_py3.pl')
keyword2idx = keyword_vocabulary['key2idx']

for s_keyword in keyword2idx:
    if s_keyword in pretrained_vocabulary:
        continue
    else:
        print("{} is not in google w2v selected vocabulary".format(s_keyword))