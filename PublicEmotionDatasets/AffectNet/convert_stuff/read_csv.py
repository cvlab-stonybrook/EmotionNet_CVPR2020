# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 19/Mar/2019 14:19
import os
import csv

emotion2idx = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "surprise": 3,
    "fear": 4,
    "disgust": 5,
    "anger": 6
}
split = "training"
annotation_file = os.path.join('/home/zwei/datasets/tarPublicEmotion/AffectNet/', '{}.csv'.format(split))

collected_data = []
with open(annotation_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            file_name = row[0]
            emotion_category = row[-3]
            # if emotion_category  7:
            collected_data.append([file_name, emotion_category])


            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            # line_count += 1
    print(f'Processed {line_count} lines.')

selected_data = []
for s_data in collected_data:
    if int(s_data[1])>6:
        continue
    else:
        selected_data.append([s_data[0], int(s_data[1])])
from PyUtils.pickle_utils import save2pickle

save2pickle('/home/zwei/Dev/AttributeNet3/PublicEmotionDatasets/AffectNet/{}.pkl'.format(split), selected_data )
print("DEB")