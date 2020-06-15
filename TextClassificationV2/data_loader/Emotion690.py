from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import string_list2dict, get_key_sorted_dict
from AdobeStockTools.TagUtils import has_digits, keepGoodTags
from collections import Counter
import tqdm
import numpy as np
import random

import os, sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)


random.seed(0)

predefined_emotions = sorted(['surprise(negative)', 'happiness', 'sadness', 'disgust', 'surprise(positive)', 'neutral',
                              'anger', 'fear'])
idx2emotion, emotion2idx = string_list2dict(predefined_emotions)




from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url

def convert_list2multihot(label_list, n_classes):
    multi_hot = np.zeros(n_classes)
    for s_label in label_list:
        multi_hot[s_label] = 1
    return multi_hot

def read_690_complete_mtrain_mtest(data_path=None, subset_N=None):

    if data_path is None:
        data_path = os.path.join(project_root, 'AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')
    annotated_data = loadpickle(data_path)

    idx2tag = loadpickle(os.path.join(project_root, 'AdobeStockSelection/EmotionNetFinal/tag2idx.pkl'))['idx2tag']
    # predefined_vocabularies = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/params/selftrained_extracted_w2v_wordnet_synsets_py3.pl')

    data = []
    if subset_N is None:
        subset = annotated_data
    else:
        subset = annotated_data[:subset_N]

    for s_data in tqdm.tqdm(subset, desc="Processing Annotated Data"):

        s_image_cid = int(get_image_cid_from_url(s_data[0], location=1))
        raw_tags =  [idx2tag[x] for x in s_data[2]]

        data.append([raw_tags, s_data[1], s_image_cid])

    random.seed(0)
    random.shuffle(data)

    dev_idx = 5000
    val_data = data[:dev_idx]
    train_data = data[dev_idx:]



    return train_data, val_data


def read_690_complete_mtrain_mtest_wo_emotion(data_path=None, subset_N=None):

    if data_path is None:
        data_path = os.path.join(project_root, 'AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')
    annotated_data = loadpickle(data_path)
    emotion2idx = loadpickle(os.path.join(project_root, 'AdobeStockSelection/EmotionNetFinal/etag2idx.pkl'))['key2idx']
    idx2tag = loadpickle(os.path.join(project_root, 'AdobeStockSelection/EmotionNetFinal/tag2idx.pkl'))['idx2tag']
    # predefined_vocabularies = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/params/selftrained_extracted_w2v_wordnet_synsets_py3.pl')

    data = []
    if subset_N is None:
        subset = annotated_data
    else:
        subset = annotated_data[:subset_N]

    for s_data in tqdm.tqdm(subset, desc="Processing Annotated Data"):

        s_image_cid = int(get_image_cid_from_url(s_data[0], location=1))
        raw_tags =  [idx2tag[x] for x in s_data[2]]
        updated_tags = []
        for s_tag in raw_tags:
            if s_tag not in emotion2idx:
                updated_tags.append(s_tag)

        data.append([updated_tags, s_data[1], s_image_cid])

    random.seed(0)
    random.shuffle(data)

    dev_idx = 2000
    val_data = data[:dev_idx]
    train_data = data[dev_idx:]



    return train_data, val_data



if __name__ == '__main__':
    # TODO: compute the tag length distribution

    from PyUtils.dict_utils import get_key_sorted_dict
    train_set, val_set = read_690_complete_mtrain_mtest()
    tag_lengths = {}
    for s_data in train_set:
        x_length = len(s_data[0])
        if x_length in tag_lengths:
            tag_lengths[x_length] += 1
        else:
            tag_lengths[x_length] = 1
    tag_lengths = get_key_sorted_dict(tag_lengths, reverse=False)
    for s_len in tag_lengths:
        print("{}\t{}".format(s_len, tag_lengths[s_len]))
    print("DB")