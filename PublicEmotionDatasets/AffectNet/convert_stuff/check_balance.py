# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 14:34


from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm

x_data = loadpickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/validation.pkl')

categories = {}

for x in tqdm.tqdm(x_data):
    if x[1] in categories:
        categories[x[1]] += 1
    else:
        categories[x[1]] = 1

print("DB")
