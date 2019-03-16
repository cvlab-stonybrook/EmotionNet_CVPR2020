# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 13/Mar/2019 21:23


from PyUtils.pickle_utils import loadpickle, save2pickle

split_files = ['/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/key_labels/CNNsplit_key_labels_test_dict.pkl',
               '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/key_labels/CNNsplit_key_labels_val_updated_dict.pkl',
               '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/key_labels/CNNsplit_key_labels_train_dict.pkl']


complete_cids = []
for s_file in split_files:
    x = loadpickle(s_file)
    for x_cid in x:
        complete_cids.append(x_cid)

assert len(set(complete_cids)) == len(complete_cids)
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/full_labels/CID_list.pkl', complete_cids)

