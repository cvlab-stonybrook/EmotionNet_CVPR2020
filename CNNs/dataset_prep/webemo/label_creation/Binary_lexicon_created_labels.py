# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): based on Jianming's code
# Email: hzwzijun@gmail.com
# Created: 22/Oct/2018 22:12

import torch
from PyUtils.pickle_utils import loadpickle, save2pickle
from GloVeTrain.models import NRCPredict
from CNNs.models.resnet import load_state_dict
from CNNs.dataset_prep.webemo import constants
from torchnet.meter.mapmeter import mAPMeter
import tqdm
from GloVeTrain.models.utils import glove_embedding_loader
from CNNs.dataset_prep.webemo.selected_webemo_wrong_labels import wrong_PN_tags
wrong_PN_tags = set(wrong_PN_tags)
from params.NRC.loaders import get_NRC_emotional_strict
from AdobestockTools.utils import word_hasAdjective
from nltk.corpus import wordnet
import GloVeTrain.word_sentiment_prediction_utils as pred_utils
NRCDict = get_NRC_emotional_strict()
from JM.modules import Net
import csv
import pandas as pd
import numpy as np
from embeddings import GloveEmbedding
# g = GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True, default='none')
import torch.nn.functional as F
from JM.constants import *
from JM.utils import get_lexicons_compiled_dict
import nltk
from JM.eval_nlp_webemo_lexicon_only import get_emotion_dict, filtered_words, count_words




def main():
    sentiment_mAP = mAPMeter()

    emotion_word_dict = get_emotion_dict()


    split = 'test'
    raw_data = loadpickle(
        '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/{}-CIDs-fullinfo.pkl'.format(split))
    inconsistent_count = 0
    # with open('JM_difference-verbose-{}-lexicon.txt'.format(split), 'w') as of_:
    image_idx = 0
    category_counts = np.zeros(3)
    image_annotations = []
    for s_data in tqdm.tqdm(raw_data, desc="Scanning"):
        RAMs_predict = s_data[5]
        gt_sentiment = np.zeros((1, 2))
        gt_sentiment[0, RAMs_predict] = 1

        p_print_lines = []
        p_print_lines.append("*** {}|{} CID: {}\n".format(image_idx, len(raw_data), s_data[-1]))

        total_labels = s_data[1]
        title = s_data[2]
        p_print_lines.append('Title: {}\n'.format(title))



        pred_sentiment = np.zeros((1, 2))
        p_counted_labels = []
        selected_labels = []

        if 'kid' in total_labels and 'baby' in total_labels:
            total_labels.remove('kid')

        pred_sentiment, p_counted_labels, selected_labels = count_words(total_labels, emotion_word_dict, filtered_words,
                                                                        selected_labels, pred_sentiment,
                                                                        p_counted_labels, subjectivity='strong')

        if pred_sentiment[0, 0] == pred_sentiment[0, 1]:
            pred_sentiment, p_counted_labels, selected_labels = count_words(nltk.word_tokenize(title),
                                                                            emotion_word_dict, filtered_words,
                                                                            selected_labels, pred_sentiment,
                                                                            p_counted_labels, subjectivity='strong')

        if pred_sentiment[0, 0] == pred_sentiment[0, 1]:
            pred_sentiment, p_counted_labels, selected_labels = count_words(total_labels, emotion_word_dict,
                                                                            filtered_words,
                                                                            selected_labels, pred_sentiment,
                                                                            p_counted_labels, subjectivity='weak')

        if pred_sentiment[0, 0] == pred_sentiment[0, 1]:
            pred_sentiment, p_counted_labels, selected_labels = count_words(nltk.word_tokenize(title),
                                                                            emotion_word_dict, filtered_words,
                                                                            selected_labels, pred_sentiment,
                                                                            p_counted_labels, subjectivity='weak')

        if pred_sentiment[0, 0] == pred_sentiment[0, 1]:
            image_annotations.append((s_data[-1], -1))
            category_counts[0] += 1
        elif  pred_sentiment[0, 0] > pred_sentiment[0, 1]:
            image_annotations.append((s_data[-1], 0))
            category_counts[1] += 1
        else:
            image_annotations.append((s_data[-1], 1))
            category_counts[2] += 1

    print("Undecided: {}, Negative: {}, Positive: {}".format(*category_counts))
    save2pickle('{}_lexion_created_strong_label_strong_title_weak_label_weak_title.pkl'.format(split), image_annotations)



if __name__ == '__main__':
    main()














