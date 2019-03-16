# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 15/Mar/2019 23:21


import matplotlib.pyplot as plt
import numpy as np
import math
from PyUtils.pickle_utils import loadpickle

emotion_name_counts = loadpickle('/home/zwei/Dev/AttributeNet3/SrockEmotionReports/emotion_name_counts.pkl')
values = np.array([emotion_name_counts[key] for key in emotion_name_counts])

names = [key for key in emotion_name_counts]

values = values[1:]
names = names[1:]
every_n = 10
names_very_n = []
for idx, s_name in enumerate(names):
    if idx % every_n == 0:
        names_very_n.append(s_name)

x = np.arange(len(names_very_n))
plt.figure(figsize=(30,8))

plt.bar(x, height=values[::every_n])
plt.xticks(x, names_very_n)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=20)
plt.savefig('Distribution.pdf')
# plt.show()