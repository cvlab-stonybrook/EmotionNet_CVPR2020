# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): convert image list to image dict keyed by image-cid
# Email: hzwzijun@gmail.com
# Created: 19/Feb/2019 11:53

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm, os
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url

src_file = 'X'
dst_file = 'X'
image_annotations = loadpickle(src_file)
image_annotation_dict = {}
for s_image_annotation in tqdm.tqdm(image_annotations, desc='converting image annotation list to dict'):
    s_image_cid = int(get_image_cid_from_url(s_image_annotation[0], 1))
    if s_image_cid not in image_annotation_dict:
        image_annotation_dict[s_image_cid] = s_image_annotation
    else:
        print("{} Double Counted".format(s_image_cid))

save2pickle(dst_file, image_annotation_dict)