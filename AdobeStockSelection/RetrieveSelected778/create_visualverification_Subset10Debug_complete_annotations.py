# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 17/Feb/2019 12:42

import tqdm
from AdobeStockTools.TagUtils import remove_hat
from PyUtils.pickle_utils import loadpickle, save2pickle
import glob, os
from AdobeStockSelection.RetrieveSelected778.utils import convert_imagename2imagerelpath
from PyUtils.json_utils import load_json_list
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url

split_data_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/CNNsplit_train.pkl')
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
searchkeys = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/tag-idx-conversion.pkl')['key2idx']
valid_tags = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/Word2Vecs/vocabularies_complete_dict_from_train.pkl')['word2idx']
cid_list = []
for s_data in split_data_list:
    s_cid = get_image_cid_from_url(s_data[0], location=1)
    cid_list.append(int(s_cid))

cid_set = set(cid_list)

processed_cids = set()
search_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_keyword_annotations.pkl')
image_locations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/imagename_cid_correspondences.pkl')
image_information = []
image_directory = '/home/zwei/datasets/stockimage_742/images-256'
image_urls = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_image_urls.pkl')
for s_raw_annotation_file in tqdm.tqdm(raw_annotation_files, desc="Iterating Through CID Files", total=len(raw_annotation_files)):


    s_raw_annotations = load_json_list(s_raw_annotation_file)

    for s_raw_annotation in s_raw_annotations:
        s_cid = s_raw_annotation['cid']
        if s_cid not in cid_set:
            continue
        if s_cid in processed_cids:
            continue
        else:
            processed_cids.add(s_cid)
            s_image_location = image_locations[s_cid]
            s_image_searchkey = search_annotations[s_cid]
            s_debug_search_key = []
            for s_searchkey in s_image_searchkey:
                if s_searchkey in searchkeys:
                    s_debug_search_key.append(s_searchkey)

            s_image_tags = remove_hat(s_raw_annotation['tags'])
            s_image_url = image_urls[s_cid]
            s_valid_tags = []
            for s_tag in s_image_tags:
                if s_tag in valid_tags:
                    s_valid_tags.append(s_tag)

            s_image_relpath = convert_imagename2imagerelpath(s_image_location)
            if not os.path.exists(os.path.join(image_directory, s_image_relpath)):
                print("Something wong wtih {}".format(s_image_relpath))
            image_information.append([s_cid, s_image_relpath, s_debug_search_key, s_valid_tags, s_image_url])

save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/debug_subset_10/visualverification_information_train.pkl', image_information)

