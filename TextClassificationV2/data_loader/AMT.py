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

    predefined_vocabularies = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/params/selftrained_extracted_w2v_wordnet_synsets_py3.pl')

    data = []
    for s_image_cid in tqdm.tqdm(annotated_data, desc="Processing Annotated Data"):
        s_data = annotated_data[s_image_cid]
        s_image_emotions = []
        for x in s_data['image_emotion']:
            s_image_emotions.extend(x)
        s_image_emotions = Counter(s_image_emotions)

        goodtags = []
        raw_tags = s_data['tags']
        for s_raw_tag in raw_tags:
            if s_raw_tag in predefined_vocabularies:
                goodtags.append(s_raw_tag)
        if len(goodtags)<1:
            continue
        data.append([goodtags, s_image_emotions, s_image_cid])

    random.seed(0)
    random.shuffle(data)

    dev_idx = len(data) // 10
    val_data = data[:dev_idx]
    train_data = data[dev_idx:]

    updated_train_data = []
    for s_data in train_data:
        goodtags, s_emotion_counter, s_image_cid = s_data

        s_emotion_label = counter2multilabel(s_emotion_counter, emotion2idx)

        updated_train_data.append([goodtags, s_emotion_label, s_image_cid])

    updated_val_data = []
    for s_data in val_data:
        goodtags, s_emotion_counter, s_image_cid = s_data
        s_emotion_label = counter2multilabel(s_emotion_counter, emotion2idx)
        updated_val_data.append([goodtags, s_emotion_label, s_image_cid])

    print("Train: {}\tVal: {}".format(len(updated_train_data), len(updated_val_data)))

    return updated_train_data, updated_val_data



def read_AMT_complete_mtrain_stest(data_path=None):

    if data_path is None:
        data_path = '/home/zwei/Dev/AttributeNet3/MturkCollectedData/data/mturk_annotations.pkl'
    annotated_data = loadpickle(data_path)

    predefined_vocabularies = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/params/selftrained_extracted_w2v_wordnet_synsets_py3.pl')

    data = []
    for s_image_cid in tqdm.tqdm(annotated_data, desc="Processing Annotated Data"):
        s_data = annotated_data[s_image_cid]
        s_image_emotions = []
        for x in s_data['image_emotion']:
            s_image_emotions.extend(x)
        s_image_emotions = Counter(s_image_emotions)

        goodtags = []
        raw_tags = s_data['tags']
        for s_raw_tag in raw_tags:
            if s_raw_tag in predefined_vocabularies:
                goodtags.append(s_raw_tag)
        if len(goodtags)<1:
            continue
        data.append([goodtags, s_image_emotions, s_image_cid])

    random.seed(0)
    random.shuffle(data)
    dev_idx = len(data) // 10
    val_data = data[:dev_idx]
    train_data = data[dev_idx:]

    updated_train_data = []
    for s_data in train_data:
        goodtags, s_emotion_counter, s_image_cid = s_data
        image_emotion_distributes = np.zeros(len(emotion2idx))
        for x in s_emotion_counter.most_common():
            image_emotion_distributes[emotion2idx[x[0]]] = x[1]

        image_emotion_distributes /= sum(image_emotion_distributes)

        updated_train_data.append([goodtags, image_emotion_distributes, s_image_cid])

    updated_val_data = []
    for s_data in val_data:
        goodtags, s_emotion_counter, s_image_cid = s_data
        if s_emotion_counter.most_common()[0][1] < 3:
            continue
        else:

            updated_val_data.append([goodtags, emotion2idx[s_emotion_counter.most_common()[0][0]], s_image_cid])


    print("Train: {}\tVal: {}".format(len(updated_train_data), len(updated_val_data)))

    return updated_train_data, updated_val_data



if __name__ == '__main__':
    # TODO: compute the tag length distribution

    from PyUtils.dict_utils import get_key_sorted_dict
    train_set, val_set = read_AMT_complete_mtrain_mtest()
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