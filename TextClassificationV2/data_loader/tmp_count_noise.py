# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): this is simply a copy of AMT and used to check for the noise in emotion labels
# Email: hzwzijun@gmail.com
# Created: 10/Mar/2019 22:44


from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import string_list2dict, get_key_sorted_dict
from AdobeStockTools.TagUtils import has_digits, keepGoodTags
from collections import Counter
import tqdm
import numpy as np
import random


random.seed(0)

predefined_emotions = sorted(['surprise(negative)', 'happiness', 'sadness', 'disgust', 'surprise(positive)', 'neutral',
                              'anger', 'fear'])
idx2emotion, emotion2idx = string_list2dict(predefined_emotions)


def counter2multilabel(counter, emotion2idx):
    emotion_vector = np.zeros(len(emotion2idx))
    for x in counter.most_common():
        emotion_vector[emotion2idx[x[0]]] = x[1]

    emotion_vector /= sum(emotion_vector)*1.
    return emotion_vector


def counter2singlelabel(counter, emotion2idx, thres=3):
    if counter.most_common()[0][1] < thres:
        return None
    else:
        return emotion2idx[counter.most_common()[0][0]]


def read_AMT_complete_mtrain_mtest(data_path=None):

    if data_path is None:
        data_path = '/home/zwei/Dev/AttributeNet3/MturkCollectedData/data/mturk_annotations.pkl'
    annotated_data = loadpickle(data_path)

    for idx, s_image_cid in enumerate(annotated_data):
        if idx > 1000:
            break
        s_data = annotated_data[s_image_cid]
        print("{}\t{}\t{}".format(idx, len(annotated_data), len(s_data['emotion-tags'])))
        print("{}".format(s_data['image_url']))
        print(', '.join(s_data['emotion-tags']))




if __name__ == '__main__':
    # TODO: compute the tag length distribution


    read_AMT_complete_mtrain_mtest()

    print("Done")