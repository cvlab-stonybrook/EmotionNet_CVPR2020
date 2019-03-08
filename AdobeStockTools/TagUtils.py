# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 28/Nov/2018 15:11

from nltk.corpus import wordnet

def remove_hat(raw_tags):
    new_tags = []
    for s_tag in raw_tags:
        s_tag = s_tag.split('^')[0]
        new_tags.append(s_tag)
    return new_tags


def has_digits(inputString):
    return any(char.isdigit() for char in inputString)


def is_good_tag(s_tag):
    if len(s_tag) < 3 or has_digits(s_tag) or len(wordnet.synsets(s_tag)) < 1:
        return False
    else:
        return True


def keepGoodTags(tag_list):
    good_tags = []
    for s_tag in tag_list:
        if len(s_tag)<3 or has_digits(s_tag):
            continue
        if len(wordnet.synsets(s_tag)) < 1:
            continue
        good_tags.append(s_tag)
    return good_tags