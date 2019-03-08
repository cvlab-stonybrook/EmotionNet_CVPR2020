# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): create train/val/test file for dataloaders in CNN
# Email: hzwzijun@gmail.com
# Created: 14/Feb/2019 10:53

from PyUtils.pickle_utils import loadpickle,save2pickle
from PyUtils.file_utils import get_stem
import os
import tqdm

data_split='test'
tag_idx_conversaiton = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag-idx-conversion.pkl')
tag_dict = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/tag_frequencies_selected.pkl')
split_cids = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_CIDs_742_{}.pkl'.format(data_split))
imagename_cid_correspondences = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/imagename_cid_correspondences.pkl')
image_tag_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_keyword_annotations.pkl')
tag2idx = tag_idx_conversaiton['key2idx']
files_per_directory = 5000
split_data_list = []
avg_tags_perimage = 0
for s_image_cid in tqdm.tqdm(split_cids, desc="Progress"):
    s_image_name = imagename_cid_correspondences[s_image_cid]
    s_image_directory_id = int(get_stem(s_image_name).split('_')[0]) // files_per_directory
    s_image_partial_path = os.path.join('{:04d}'.format(s_image_directory_id), s_image_name)
    s_image_tags = image_tag_annotations[s_image_cid]
    s_labels = []
    avg_tags_perimage += len(s_image_tags)
    for s_image_tag in s_image_tags:
        s_labels.append(tag2idx[s_image_tag])
    split_data_list.append([s_image_partial_path, s_labels])
save2pickle('data_v2/CNNsplit_{}.pkl'.format(data_split), split_data_list)
print("Average tags per image: {}, total: {} images".format(avg_tags_perimage*1./len(split_cids), len(split_cids)))
print("Done")