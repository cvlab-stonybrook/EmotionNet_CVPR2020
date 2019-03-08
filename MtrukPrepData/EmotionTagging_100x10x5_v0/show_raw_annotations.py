from PyUtils.pickle_utils import loadpickle
from Datasets.WebEmo.constants import labels as webemo_labels

webemo_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/Datasets/WebEmo_AMT_Prep/data/test_url_tags_emotion25.pkl')
mturk_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_100x10x5_v0/data/results_imagebased.txt.pkl')
raw_annotations = []
category_02_count = {}
category_06_count = {}
category_25_count = {}

for s_image_cid in mturk_annotations:
    s_image_mturk_annotation = mturk_annotations[s_image_cid]
    s_raw_annotations = webemo_annotations[s_image_cid][2]
    for s_raw_annotation in s_raw_annotations:
        if webemo_labels[s_raw_annotation][0] in category_25_count:
            category_25_count[webemo_labels[s_raw_annotation][0]] += 1
        else:
            category_25_count[webemo_labels[s_raw_annotation][0]] = 1
        if webemo_labels[s_raw_annotation][1] in category_06_count:
            category_06_count[webemo_labels[s_raw_annotation][1]] += 1
        else:
            category_06_count[webemo_labels[s_raw_annotation][1]] = 1

        if webemo_labels[s_raw_annotation][2] in category_02_count:
            category_02_count[webemo_labels[s_raw_annotation][2]] += 1
        else:
            category_02_count[webemo_labels[s_raw_annotation][2]] = 1

print("DB")
