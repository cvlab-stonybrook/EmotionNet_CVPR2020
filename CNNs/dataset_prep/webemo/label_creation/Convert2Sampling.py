# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 31/Oct/2018 20:48

from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm

data = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/Emotion6-train-w2v-NN5.pkl')


Categorical_CIDs = {}
Categorical_CID_counts = {}
for s_data in tqdm.tqdm(data):
    # s_CID = get_stem(s_data[0])
    s_rel_path = s_data[0]
    # s_CID = int(get_stem(s_rel_path))

    s_new_label = s_data[1]

    if s_new_label in Categorical_CIDs:
        Categorical_CIDs[s_new_label].append(s_rel_path)
        Categorical_CID_counts[s_new_label] += 1
    else:
        Categorical_CIDs[s_new_label] = [s_rel_path]
        Categorical_CID_counts[s_new_label] = 1

for s_id in Categorical_CID_counts:
    assert len(Categorical_CIDs[s_id]) == Categorical_CID_counts[s_id], "ERROR"

data = {'categories': Categorical_CIDs, 'counts': Categorical_CID_counts}
save2pickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/data/Emotion6-train-sample-w2v-NN5.pkl', data)