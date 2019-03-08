# create 100 ekman's emotion
# we are using ekman's 6 emotion

from Datasets.WebEmo.constants import labels as emotion_25
from PyUtils.pickle_utils import loadpickle, save2pickle
webemo_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/Datasets/WebEmo_AMT_Prep/data/test_url_tags_emotion25.pkl')
target_categories_distributions = {'horror': 100, 'nervousness': 100, 'irritabilty': 100, 'rage': 100, 'disgust': 200, 'envy': 100, 'surprise': 300}
previous_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_100x10x5_v0/data/results_imagebased.txt.pkl')
selected_categories_distributions = {'horror': 0, 'nervousness': 0, 'irritabilty': 0, 'rage': 0, 'disgust': 0, 'envy': 0, 'surprise': 0}

webemo_cid_list = [x for x in webemo_annotations]
selected_list = []
for s_cid in webemo_cid_list:
    if s_cid in previous_annotations:
        continue
    current_emotion = emotion_25[webemo_annotations[s_cid][2][0]][0]
    if current_emotion in selected_categories_distributions:
        if selected_categories_distributions[current_emotion] < target_categories_distributions[current_emotion]:
            selected_categories_distributions[current_emotion] += 1
            selected_list.append(webemo_annotations[s_cid])

save2pickle('emotiontagging_1000_v2.pkl', selected_list)



print("DEB")
