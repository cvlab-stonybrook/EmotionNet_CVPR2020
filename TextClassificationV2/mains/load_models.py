# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 13/Mar/2019 19:56
import torch
from TextClassificationV2.models.TextCNN import TextCNN_NLT
from CNNs.models.resnet import load_state_dict
from PyUtils.pickle_utils import loadpickle







if __name__ == '__main__':

    ckpt_file = '/home/zwei/Dev/AttributeNet3/TextClassificationV2/ckpts/TextCNN_googlenews_NLT_Static.pth.tar'

    ckpt_info = torch.load(ckpt_file)

    args_model = ckpt_info['args_model']
    text_model = TextCNN_NLT(args_model)

    load_state_dict(text_model, ckpt_info['state_dict'])

    x_weight = text_model.embedding.weight[1]

    args_data = ckpt_info['args_data']
    embed_dict = loadpickle('/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_dicts/googlenews_S_w2v_dict.pl')
    x_weight_from_data = embed_dict[args_data.idx2tag[1]]
    print("Done")