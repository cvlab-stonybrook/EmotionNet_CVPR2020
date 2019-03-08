# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 24/Feb/2019 17:08

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
from PyUtils.dict_utils import get_key_sorted_dict
previously_labeled = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/AMT_data/processed/v1v2_withCIDs.pkl')
current_candidate = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_val_dict.pkl')
new_candidate = {}
for s_candidate in tqdm.tqdm(current_candidate):
    if s_candidate in previously_labeled:
        continue
    else:
        new_candidate[s_candidate] = current_candidate[s_candidate]

new_candidate = get_key_sorted_dict(new_candidate, reverse=False)
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_val_webemo_annotation_excluded_dict.pkl', new_candidate)
print("DEB")