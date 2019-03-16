# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 14/Mar/2019 17:40

import json
from difflib import SequenceMatcher
from string import digits
import numpy
import string
import unicodedata
import re
import collections
from itertools import compress
import operator
import itertools

def most_common(lst):
    return max(set(lst), key=lst.count)



with open('/mnt/ilcompf8d1/user/rapanda/datasets/advt_dataset/annotations_images/image/Sentiments.json', 'r') as f:
    data = json.load(f)


for key, value in data.items():
    data_list.write(path + key + ' ' + str(int(most_common(list(itertools.chain(*value))))-1) + '\n')


