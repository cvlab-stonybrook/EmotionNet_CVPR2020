#
# This file is part of the AttributeNet3 project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Jun-15.
# 09: 50
# All Rights Reserved
#

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
from Dataset_release.zw_utils import replace_240

annotation_file = '/home/zwei/Dev/PastProjects/AttributeNet3/Dataset_release/StockEmotion/image_download_list/image_download_list.pkl'
raw_annotations = loadpickle(annotation_file)

for s_annotation in tqdm.tqdm(raw_annotations):
    s_annotation[1] = replace_240(s_annotation[1])

save2pickle('/home/zwei/Dev/PastProjects/AttributeNet3/Dataset_release/StockEmotion/image_download_list'
            '/dataset_image_urls_240.pkl', raw_annotations)

print("Done")
