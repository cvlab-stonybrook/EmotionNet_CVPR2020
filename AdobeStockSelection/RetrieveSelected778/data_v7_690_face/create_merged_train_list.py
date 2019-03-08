# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 06/Mar/2019 13:54

from PyUtils.pickle_utils import loadpickle, save2pickle, loadpickle_python2_compatible
import tqdm
import os
previous_annotations_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_tag_labels_train_dict.pkl')
face_image_list = loadpickle_python2_compatible('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v7_690_face/face_crop_single_index.pkl')

updated_list = []

for s_cid in tqdm.tqdm(previous_annotations_dict, total=len(previous_annotations_dict)):
    x_annotation = previous_annotations_dict[s_cid]
    updated_list.append(x_annotation)

for s_face_crop in tqdm.tqdm(face_image_list, total=len(face_image_list), desc="adding face crops into training list"):
    s_face_cid = int(os.path.basename(s_face_crop[0]).split('_')[1])
    if s_face_cid in previous_annotations_dict:
        s_tag_ids = previous_annotations_dict[s_face_cid][1]
        updated_list.append([s_face_crop[0], s_tag_ids])


print("Total {} images for training".format(len(updated_list)))
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v7_690_face/CNNsplit_tag_labels_train+face.pkl', updated_list)
