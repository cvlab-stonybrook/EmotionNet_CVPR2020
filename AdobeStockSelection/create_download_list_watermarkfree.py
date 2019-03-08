# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 06/Mar/2019 10:36

from PyUtils.pickle_utils import loadpickle, save2pickle
import os
import tqdm

full_reso_list = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/image_download_list.pkl')

no_warter_mark_list = []

for s_image_location in tqdm.tqdm(full_reso_list):
    s_rel_path = s_image_location[0]
    s_url = s_image_location[1]
    s_no_warter_mark_url_image_name = os.path.join(os.path.dirname(s_url), '220'+os.path.basename(s_url)[4:])
    no_warter_mark_list.append([s_rel_path, s_no_warter_mark_url_image_name])


save2pickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/image_download_nowatermark_list.pkl', no_warter_mark_list)