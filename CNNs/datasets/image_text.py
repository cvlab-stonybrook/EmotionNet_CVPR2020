# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 03/Mar/2019 10:12
import os
import torch
from CNNs.dataloaders.image_text_loader import ImageTextRelLists
from CNNs.dataloaders.transformations import *
from PyUtils.pickle_utils import loadpickle
import numpy as np



def label_text_textembd_transform(args):
    text_transform = convert2txtidANDembed(args.text_embed, args.tag2clsidx, args.idx2tag, args.word_dim, args.vocab_size, args.max_length)
    label_transform = multilabel2multihot(args.num_classes)
    return label_transform, text_transform

def label_embed_transform(args):
    text_transform = convert2embed(args.text_embed, args.idx2tag, args.word_dim)
    label_transform = multilabel2multihot(args.num_classes)
    return label_transform, text_transform


def convert2txtidANDembed(text_embed, tag2clsidx, idx2tag, word_dim, vocab_size, max_lengh=100):
    def target_transform(sentence):

        sentence_length = len(sentence)
        sentence_words = []
        for s_tag_id in sentence:
            if s_tag_id in idx2tag:
                sentence_words.append(idx2tag[s_tag_id])

        sentence_clsinput = []
        for s_word in sentence_words:
            if s_word in tag2clsidx:
                sentence_clsinput.append(tag2clsidx[s_word])

        if len(sentence_clsinput) >= max_lengh:
            sentence_idx = sentence_clsinput[:max_lengh]
        else:
            sentence_idx = sentence_clsinput + [vocab_size+1]* (max_lengh - len(sentence_clsinput))

        sentence_idx =torch.LongTensor(sentence_idx)

        sentence_embeddings = []
        for s_tag in sentence_words:
                if s_tag in text_embed:
                    sentence_embeddings.append(text_embed[s_tag])

        if len(sentence_embeddings) == 0:
            avg_embedding = np.random.randn(word_dim)
        else:
            avg_embedding = np.mean(np.array(sentence_embeddings), axis=0)
        avg_embedding =torch.FloatTensor(avg_embedding)

        return sentence_idx, avg_embedding
    return target_transform


def convert2embed(text_embed, idx2tag, word_dim):
    def target_transform(sentence):
        embeddings = []
        for s_tag_id in sentence:
            if s_tag_id in idx2tag:
                s_tag = idx2tag[s_tag_id]
                if s_tag in text_embed:
                    embeddings.append(text_embed[s_tag])

        if len(embeddings) == 0:
            avg_embedding = np.random.randn(word_dim)
        else:
            avg_embedding = np.mean(np.array(embeddings), axis=0)
        avg_embedding =torch.FloatTensor(avg_embedding)

        return avg_embedding
    return target_transform



def label_text_transform(n_classes, vocab_size, max_length):
    text_transform = padd_sentences(vocab_size, max_length)
    label_transform = multilabel2multihot(n_classes)
    return label_transform, text_transform

def padd_sentences(vocab_size=1000, max_lengh=100):
    def target_transform(sentence):
        sentence_length = len(sentence)

        if sentence_length >= max_lengh:
            new_sentence = sentence[:max_lengh]
        else:
            new_sentence = sentence + [vocab_size+1]* (max_lengh - sentence_length)

        new_sentence =torch.LongTensor(new_sentence)

        return new_sentence
    return target_transform


def multilabel2multihot(n_classes=500):
    def target_transform(label_list):
        label_vector = np.zeros(n_classes, dtype=np.float)
        for s_label in label_list:
            label_vector[s_label] = 1
        label_vector /= (len(label_list)*1.0)
        return torch.FloatTensor(label_vector)
    return target_transform


def image_text_train(args):
    #FIXME:
    # annotation_file = annotation_file.format('train')
    image_information = loadpickle(args.train_file)
    dataset = ImageTextRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=label_text_transform(args.num_classes, args.vocab_size, args.sent_len))
    return dataset


def image_textembed_train(args):
    #FIXME:
    # annotation_file = annotation_file.format('train')
    image_information = loadpickle(args.train_file)
    dataset = ImageTextRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=label_embed_transform(args))
    return dataset


def image_cls_text_textembed_train(args):
    #FIXME:
    # annotation_file = annotation_file.format('train')
    image_information = loadpickle(args.train_file)
    dataset = ImageTextRelLists(image_paths=image_information, image_root=args.data_dir, transform=get_train_simple_transform(), target_transform=label_text_textembd_transform(args))
    return dataset


if __name__ == '__main__':
    from argparse import Namespace
    from CNNs.dataloaders.utils import none_collate

    # args = Namespace(num_classes=690, vocab_size=1000, sent_len=100)
    # args.train_file = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_tag_label_tag_full_train.pkl'
    # args.data_dir = '/home/zwei/datasets/stockimage_742/images-256'
    # dataset = image_text_train(args)
    # val_loader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=10, shuffle=False,
    #                                          num_workers=4, pin_memory=True, collate_fn=none_collate)
    # import tqdm
    #
    # for s_images, s_labels, s_text in tqdm.tqdm(val_loader):
    #     pass
    #
    # print("Done")
    import tqdm

    args = Namespace(num_classes=690, vocab_size=1000, sent_len=100, text_ckpt="/home/zwei/Dev/AttributeNet3/TextClassificationV2/ckpts/TextCNN_googlenews_690Removed_NLTDAN_1M_Static.pth.tar",
                     idx2tag='/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/tag2idx.pkl',
                     text_embed='/home/zwei/Dev/AttributeNet3/LanguageData/word2vec_dicts/googlenews_S_w2v_dict.pl',
                     train_file="/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl",
                     data_dir="/home/zwei/datasets/stockimage_742/images-256")
    args.word_dim = 300
    args.max_length = 100
    text_data = torch.load(args.text_ckpt, map_location=lambda storage, loc: storage)

    args.tag2clsidx = text_data['args_data'].tag2idx
    args.vocab_size = len(args.tag2clsidx)

    args.text_embed = loadpickle(args.text_embed)
    args.idx2tag = loadpickle(args.idx2tag)['idx2tag']
    dataset = image_cls_text_textembed_train(args)
    for s_images, s_labels, s_text in tqdm.tqdm(dataset):
        pass
