# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 17/Mar/2019 18:14



from PyUtils.pickle_utils import loadpickle
import tqdm
import os
import numpy as np
train_data = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/EmotionNetFinal/CNNsplit_tag_labels+full_tagidx_train+face.pkl')



co_mat = np.zeros([690, 690])

for s_data in tqdm.tqdm(train_data):
    s_name = s_data[0]
    s_name_parts = s_name.split(os.sep)
    if len(s_name_parts) == 2:
        s_emotion = s_data[1]
        for i in range(len(s_emotion)):
            co_mat[s_emotion[i], s_emotion[i]] += 1
            for j in range(i+1, len(s_emotion)):
                co_mat[s_emotion[i], s_emotion[j]] += 1
                co_mat[s_emotion[j], s_emotion[i]] += 1



np.savetxt('comat.txt', co_mat, fmt="%d")

print("DB")