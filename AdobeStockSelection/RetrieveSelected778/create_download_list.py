# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 12/Feb/2019 16:40
import os
from PyUtils.pickle_utils import loadpickle, save2pickle
from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
image_urls = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/dataset_image_urls.pkl')
import tqdm

n_images_per_dir = 5000
imagename_correspondences = {}
image_download_list = []
for s_idx, s_image_cid in tqdm.tqdm(enumerate(image_urls), total=len(image_urls)):
    image_cid_from_url = get_image_cid_from_url(image_urls[s_image_cid])
    assert str(s_image_cid) == image_cid_from_url
    save_image_name = '{:08d}_{}.jpg'.format(s_idx, s_image_cid)
    imagename_correspondences[s_image_cid] = save_image_name
    directory = '{:04d}'.format(s_idx//n_images_per_dir)
    image_download_list.append([os.path.join(directory, save_image_name), image_urls[s_image_cid]])
save2pickle('data_v2/image_download_list.pkl', image_download_list)
save2pickle('data_v2/imagename_cid_correspondences.pkl', imagename_correspondences)

print("DB")