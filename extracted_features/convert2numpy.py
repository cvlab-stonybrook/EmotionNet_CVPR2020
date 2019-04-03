# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 20/Mar/2019 11:24

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm
import numpy as np
collected_features = loadpickle('/home/zwei/Dev/AttributeNet3/extracted_features/Desk_feature_extractor_config/StockEmotionCls+SentEmbd-20190320105206/feature.pkl')

feature_matrix = []
feature_names = []

for x_name in tqdm.tqdm(collected_features):
    feature_names.append(x_name)
    feature_matrix.append(collected_features[x_name][0])

feature_matrix_np = np.array(feature_matrix)
save2pickle('feature_numpy.pkl', {'imagenames': feature_names, 'feature': feature_matrix_np})