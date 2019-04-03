# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 20/Mar/2019 11:24

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url

type_name = 'ImageNet'

print(type_name)
if type_name == 'ImageNet':

    collected_features = loadpickle('/home/zwei/Dev/AttributeNet3/extracted_features/Desk_feature_extractor_config/ImageNet-20190320110503/feature_numpy.pkl')
else:
    collected_features = loadpickle('/home/zwei/Dev/AttributeNet3/extracted_features/Desk_feature_extractor_config/StockEmotionCls+SentEmbd-20190320105206/feature_numpy.pkl')
image_urls = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/other/dataset_image_urls.pkl')
feature_matrix = collected_features['feature']
feature_names = collected_features['imagenames']

for idx, image_name in enumerate(feature_names):
    if idx < 50000 or idx > 50500 :
        continue
    image_cid = int(get_image_cid_from_url(image_name, location=1))
    if image_cid not in image_urls:
        continue
    image_url = image_urls[image_cid]
    image_feature = feature_matrix[idx]
    distances = euclidean_distances(np.expand_dims(image_feature, 0), feature_matrix)
    distances_sort = np.argsort(distances[0])[:15:1]
    print("** {} **".format(idx))
    print("{}".format(image_url))
    for candidate_idx, candiate_position in enumerate(distances_sort[2:]):
        s_cid = int(get_image_cid_from_url(feature_names[candiate_position], location=1))
        if s_cid not in image_urls:
            continue
        s_url = image_urls[s_cid]
        print('{}\t{:.2f}\t{}'.format(candidate_idx, distances[0,candiate_position], s_url))


# for x_name in tqdm.tqdm(collected_features):
#     feature_names.append(x_name)
#     feature_matrix.append(collected_features[x_name][0])

# feature_matrix_np = np.array(feature_matrix)
# save2pickle('feature_numpy.pkl', {'imagenames': feature_names, 'feature': feature_matrix_np})