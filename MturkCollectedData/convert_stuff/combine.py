# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 28/Feb/2019 21:17

from PyUtils.pickle_utils import loadpickle, save2pickle
import glob
import os
from PyUtils.json_utils import load_json_list
import tqdm
from AdobeStockTools.TagUtils import remove_hat
val_key_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/CNNsplit_key_labels_val_dict.pkl')
raw_annotation_files = glob.glob(os.path.join('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/selected_keywords_retrieve_v2', '*.json'))
emotion_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/E_vocabulary.pkl')
pretrained_vocabulary = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/pre_extract_w2v/params/googlenews_extracted_w2v_wordnet_synsets_py3.pl')
idx2emotion = emotion_vocabulary['idx2key']
emotion2idx = emotion_vocabulary['key2idx']
turk_files = ['/home/zwei/Dev/AttributeNet3/MturkCollectedData/results_imagebased/results_imagebased.txt_v3.pkl',
              '/home/zwei/Dev/AttributeNet3/MturkCollectedData/results_imagebased/results_imagebased.txt_v4.pkl']
image_annoations = {}
for s_turk_file in turk_files:
    s_image_annotaiton = loadpickle(s_turk_file)
    for s_cid in s_image_annotaiton:
        image_annoations[s_cid] = s_image_annotaiton[s_cid]

processed_cids = set()
for s_raw_file in tqdm.tqdm(raw_annotation_files):
        raw_annotations = load_json_list(s_raw_file)
        for s_annotation in raw_annotations:
            s_cid = s_annotation['cid']
            if str(s_cid) not in image_annoations or s_cid in processed_cids:
                continue
            else:
                processed_cids.add(s_cid)
                s_tags = remove_hat(s_annotation['tags'])
                updated_tags = []
                for s_tag in s_tags:
                    if s_tag in pretrained_vocabulary:
                        updated_tags.append(s_tag)
                if len(updated_tags)>=1:
                    image_annoations[str(s_cid)]['tags'] = updated_tags
                    emotion_tags = []
                    for x in updated_tags:
                        if x in emotion2idx:
                            emotion_tags.append(x)
                    image_annoations[str(s_cid)]['emotion-tags'] = emotion_tags
                    image_annoations[str(s_cid)]['keys'] = [idx2emotion[x] for x in  val_key_annotations[s_cid][1]]
                    image_annoations[str(s_cid)]['rel-path'] = val_key_annotations[s_cid][0]
updated_image_annotations = {}
for s_cid in image_annoations:
    if 'tags' in image_annoations[s_cid]:
        updated_image_annotations[int(s_cid)] = image_annoations[s_cid]
    else:
        print("{} do not have labels in training set".format(s_cid))

print("{} were kept".format(len(updated_image_annotations)))
save2pickle('/home/zwei/Dev/AttributeNet3/MturkCollectedData/data/mturk_annotations.pkl', updated_image_annotations)

print("DB")
