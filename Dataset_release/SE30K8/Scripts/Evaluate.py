#
# This file is part of the AttributeNet3 project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Jun-15.
# 08: 51
# All Rights Reserved
#

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm


annotation_file = '/Dataset_release/SE30K8/annotations/mturk_annotations_240.pkl.keep'
raw_annotations = loadpickle(annotation_file)

# updated_annotations = {}
for s_idx, (s_key, s_item) in enumerate(tqdm.tqdm(raw_annotations.items())):
            # if s_idx % 500 == 0:
            #     print(s_item['image_url'])
        s_emotion_annotations = s_item['image_emotion']
        for s_emotion in s_emotion_annotations:
            if len(s_emotion)>1:
                print(s_key)
print("DB")
