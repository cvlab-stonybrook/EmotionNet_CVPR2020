# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Nov/2018 21:33

from PyUtils.pickle_utils import loadpickle,save2pickle

split = 'train'
file_path = '/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/New/adopted_data/{}-CIDs-fullinfo.pkl'.format(split)
data = loadpickle(file_path)

CIDs = set()
new_data = {}
for s_data in data:
    if s_data[0] in CIDs:
        print("{} Already Exists!".format(s_data[0]))
        new_data[s_data[0]][3].append(s_data[3])
        new_data[s_data[0]][4].append(s_data[4])
        new_data[s_data[0]][5].append(s_data[5])
        continue
    else:
        CIDs.add(s_data[0])
        new_data[s_data[0]] = list(s_data)
        new_data[s_data[0]][3] = [s_data[3]]
        new_data[s_data[0]][4] = [s_data[4]]
        new_data[s_data[0]][5] = [s_data[5]]


print("{} Previous, {} Non-Repeat".format(len(data), len(new_data)))
save2pickle('{}-CIDs-fullinfo-non-repeat.pkl'.format(split), new_data)