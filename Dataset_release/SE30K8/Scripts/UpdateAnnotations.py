#
# This file is part of the AttributeNet3 project.
# convert previous 1000 max image to 240 for release purpose
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Jun-15.
# 08: 43
# All Rights Reserved
#

from PyUtils.pickle_utils import loadpickle, save2pickle
from Dataset_release.zw_utils import replace_240
import tqdm

raw_annotation_file = '/home/zwei/Dev/PastProjects/AttributeNet3/Dataset_release/SE30K8/annotations/mturk_annotations' \
                     '.pkl'
save_file = '/Dataset_release/SE30K8/annotations/mturk_annotations_240.pkl.keep'
raw_annotations = loadpickle(raw_annotation_file)

# updated_annotations = {}
for s_key, s_item in tqdm.tqdm(raw_annotations.items()):
    s_item['image_url'] = replace_240(s_item['image_url'])

save2pickle(save_file, raw_annotations)

print("DB")