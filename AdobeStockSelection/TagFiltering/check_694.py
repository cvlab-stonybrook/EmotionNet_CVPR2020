# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 23/Feb/2019 10:39

from PyUtils.pickle_utils import loadpickle
from AdobeStockTools.TagUtils import is_good_tag
keyword_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl')

keyword2idx = keyword_vocabulary['key2idx']

for s_keyword in keyword2idx:
    if is_good_tag(s_keyword):
        continue
    else:
        print("{} is not a standard good tag".format(s_keyword))
