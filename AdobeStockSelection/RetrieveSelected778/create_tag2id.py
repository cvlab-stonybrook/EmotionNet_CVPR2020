# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create tag2id mapping
# Email: hzwzijun@gmail.com
# Created: 14/Feb/2019 10:56

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.dict_utils import get_value_sorted_dict, get_key_sorted_dict, string_list2dict, idx_key_conversion
tag_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')
tag_list = list(tag_dict.keys())
tag_list.sort()

key2idx, idx2key = idx_key_conversion(tag_list)
save2pickle('data_v2/tag-idx-conversion.pkl', {'idx2key': idx2key, 'key2idx': key2idx})

print("Done, saved to {}".format('data_v2/tag-idx-conversion.pkl'))
