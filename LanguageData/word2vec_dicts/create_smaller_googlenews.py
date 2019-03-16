# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create smaller google news data
# Email: hzwzijun@gmail.com
# Created: 13/Mar/2019 17:08


from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm

googlenews_dict = loadpickle('/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_dicts/googlenews_w2v_dict.pl')
selftrained_dict = loadpickle('/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_dicts/selftrained_shuffle_w2v_dict.pl')
googlenews_s_dict = {}

for s_word in tqdm.tqdm(googlenews_dict, total=len(googlenews_dict)):
    if s_word in selftrained_dict:
        googlenews_s_dict[s_word] = googlenews_dict[s_word]
print("Created a smaller set of {} compared to previous {} ".format(len(googlenews_s_dict), len(googlenews_dict)))
save2pickle('/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_dicts/googlenews_S_w2v_dict.pl', googlenews_s_dict)


