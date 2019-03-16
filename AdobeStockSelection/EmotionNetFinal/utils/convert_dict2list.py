# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): convert image list to image dict keyed by image-cid
# Email: hzwzijun@gmail.com
# Created: 19/Feb/2019 11:53





from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm, os

src_file = "/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/key_labels/CNNsplit_key_labels_val_updated_dict.pkl"
dst_file = "/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/key_labels/CNNsplit_key_labels_val_updated.pkl"
image_annotations_dict = loadpickle(src_file)
image_annotations_list = []
for s_image_cid in tqdm.tqdm(image_annotations_dict, desc='converting image annotation list to dict'):

    image_annotations_list.append(image_annotations_dict[s_image_cid])
assert len(image_annotations_list) == len(image_annotations_dict)
save2pickle(dst_file, image_annotations_list)