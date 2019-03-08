from PyUtils.pickle_utils import loadpickle
from PyUtils.dict_utils import get_value_sorted_dict
import operator
from Datasets.WebEmo.constants import labels as webemo_labels
def convert2dict(emotion_list):
    emotion_dict = {}
    for s_emotion in emotion_list:
        if s_emotion in emotion_dict:
            emotion_dict[s_emotion] += 1
        else:
            emotion_dict[s_emotion] = 1



    return get_value_sorted_dict(emotion_dict)

mturk_annotations = loadpickle('/home/zwei/Dev/AttributeNet3/MtrukPrepData/EmotionTagging_100x10x5_v0/data/results_imagebased.txt.pkl')
webemo_annotaitons = loadpickle('/home/zwei/Dev/AttributeNet3/Datasets/WebEmo_AMT_Prep/data/test_url_tags_emotion25.pkl')

mturk_annotation_distribution = {}
for s_idx, s_annotation in enumerate(mturk_annotations):
    # print(" ** {}\t{}".format(s_idx, s_annotation))

    s_webemo_annotation = webemo_annotaitons[s_annotation]
    # print( ', '.join(['{};{};{}'.format(webemo_labels[x][2], webemo_labels[x][1], webemo_labels[x][0]) for x in s_webemo_annotation[2]]) )
    all_annotations = []
    for s_emotion in mturk_annotations[s_annotation]['image_emotion']:
        all_annotations.extend(s_emotion)

    emotion_dict = convert2dict(all_annotations)
    max_key =max(emotion_dict.items(), key=operator.itemgetter(1))[0]
    max_value = emotion_dict[max_key]
    if max_value >= 3:
        if max_key in mturk_annotation_distribution:
            mturk_annotation_distribution[max_key] += 1
        else:
            mturk_annotation_distribution[max_key] = 1

    # print('; '.join(["{}({})".format(x, emotion_dict[x]) for x in emotion_dict]))

print("DB")