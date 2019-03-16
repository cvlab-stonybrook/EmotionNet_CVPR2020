# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 14/Mar/2019 21:40

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import string_list2dict
import tqdm

data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/full_labels/CID_full_tag_dict.pkl')

tag_vocab = set()
for s_cid in tqdm.tqdm(data, total=len(data)):
    x_tag_list = data[s_cid]
    for x_tag in x_tag_list:
        if x_tag in tag_vocab:
            continue
        else:
            tag_vocab.add(x_tag)

tag_vocab = sorted(list(tag_vocab))
idx2tag, tag2idx = string_list2dict(tag_vocab)
save2pickle('vocab_dict.pkl', {'tag2idx': tag2idx, 'idx2tag': idx2tag})
