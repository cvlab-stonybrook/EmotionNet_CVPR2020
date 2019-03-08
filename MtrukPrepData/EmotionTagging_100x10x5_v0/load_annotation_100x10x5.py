from PyUtils.pickle_utils import loadpickle
from EmotionTag.load_csv_annotations import load_verified
from Datasets.WebEmo import constants


image_emotion_file = '/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_100x10x5_data_v0/data/results_imagebased.txt.pkl'
image_emotions = loadpickle(image_emotion_file)

verified_emotion_file = '/home/zwei/Dev/AttributeNet3/EmotionTag/emotion_annotations_csv/Verify_20190210.csv'
verified_emotion_list, transer_dict, complete_list = load_verified(verified_emotion_file)

image_raw_tags = loadpickle('/home/zwei/Dev/AttributeNet3/Datasets/WebEmo_AMT_Prep/data/test_url_tags_emotion25.pkl')

selected_tag_set = set(verified_emotion_list)

for idx, s_image_cid in enumerate(image_emotions):
    s_image_raw_tag = image_raw_tags[s_image_cid]
    s_tags = s_image_raw_tag[1]
    s_WebEmo_categories = '; '.join(['{}({})'.format(constants.labels[i][0], constants.labels[i][1]) for i in  s_image_raw_tag[2]])
    print("**{}\t {}".format(idx,s_image_cid))
    print("{}".format(image_emotions[s_image_cid]['image_url']))
    print(image_emotions[s_image_cid]['image_emotion'])
    print("WebEmoAnnotaiton:{}".format(s_WebEmo_categories))
    selected_tags = []
    for s_tag in s_tags:
        if s_tag in selected_tag_set:
            selected_tags.append(s_tag)

    print("Selcted tags: {}".format(', '.join(selected_tags)))


print("DEB")