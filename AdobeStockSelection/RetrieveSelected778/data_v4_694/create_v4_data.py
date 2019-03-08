# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Feb/2019 10:58

from PyUtils.pickle_utils import loadpickle, save2pickle
from PyUtils.json_utils import load_json_list
from AdobeStockTools.TagUtils import remove_hat
import glob, os
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
import tqdm

data_split = 'test'
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
keyword2idx = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/tag-idx-conversion.pkl')['key2idx']

image_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v3_694/CNNsplit_{}.pkl'.format(data_split))
image_annotation_dict = {}
for s_image_annotation in tqdm.tqdm(image_annotations, desc='converting image annotation list to dict'):
    s_image_cid = int(get_image_cid_from_url(s_image_annotation[0], 1))
    if s_image_cid not in image_annotation_dict:
        image_annotation_dict[s_image_cid] = s_image_annotation
    else:
        print("{} Double Counted".format(s_image_cid))
processed_cids = set()
new_image_annotations = []
for s_raw_annotation_file in tqdm.tqdm(raw_annotation_files, desc=''):
    s_raw_annotations = load_json_list(s_raw_annotation_file)
    for s_raw_annotation in s_raw_annotations:
        s_image_cid = s_raw_annotation['cid']
        if s_image_cid in image_annotation_dict and s_image_cid not in processed_cids:
            processed_cids.add(s_image_cid)
            new_image_tags =[]
            s_image_annotation = image_annotation_dict[s_image_cid]
            s_image_raw_tags = remove_hat(s_raw_annotation['tags'])
            for s_raw_tag in s_image_raw_tags:
                if s_raw_tag in keyword2idx:
                    new_image_tags.append(keyword2idx[s_raw_tag])

            new_image_annotations.append([s_image_annotation[0], new_image_tags])
save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v4_694/CNNsplit_{}.pkl'.format(data_split), new_image_annotations)





