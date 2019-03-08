# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Oct/2018 16:53

import tqdm

def string_list2dict(input_list):
    idx2item = {}
    item2idx = {}
    for idx, item in enumerate(input_list):
        idx2item[idx] = item
        item2idx[item] = idx
    return idx2item, item2idx

#TODO: check and update later...
def get_value_sorted_dict(orig_dict, reverse=True):
    # reverse: True, high to low, False: low to high
    orig_dict = [(k, orig_dict[k]) for k in
                 sorted(orig_dict, key=orig_dict.get, reverse=reverse)]
    keyword_frequencies_dict = {}
    for s_key, s_value in orig_dict:
        keyword_frequencies_dict[s_key] = s_value
    return keyword_frequencies_dict


def get_key_sorted_dict(keyword_frequency, reverse=True):
    # reverse: True, high to low, False: low to high
    keysorted_keyword_frequency = sorted(keyword_frequency.items(), key=lambda s: s[0], reverse=reverse)
    keyword_frequencies_dict = {}
    for s_key, s_value in keysorted_keyword_frequency:
        keyword_frequencies_dict[s_key] = s_value
    return keyword_frequencies_dict


def idx_key_conversion(keywordlist):
    idx2key = {}
    key2idx = {}
    for idx, s_keyword in enumerate(keywordlist):
        idx2key[idx] = s_keyword
        key2idx[s_keyword] = idx
    return key2idx, idx2key


def key_val_conversion(orig_dict):
    new_dict = {}
    for s_key in orig_dict:
        s_item = orig_dict[s_key]
        if s_item not in new_dict:
            new_dict[s_item] = s_key
        else:
            print("{} is already in dict".format(s_item))

    return new_dict

