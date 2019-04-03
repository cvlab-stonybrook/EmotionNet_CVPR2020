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

emotion_name_counts = loadpickle('/home/zwei/Dev/AttributeNet3/StockEmotionReports/emotion_name_counts.pkl')
values = np.array([emotion_name_counts[key] for key in emotion_name_counts])

names = [key for key in emotion_name_counts]

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x * 1e-3)

values = values[1:]
names = names[1:]
every_n = 10
names_very_n = []
values_very_n = []
for idx, s_name in enumerate(names):
    if idx % every_n == 0:
        if len(s_name) > 10:
            names_very_n.append(s_name[:10]+'.')
        else:
            names_very_n.append(s_name)
        values_very_n.append(values[idx])
x = np.arange(len(names_very_n))
plt.figure(figsize=(30,8))


# fig, ax = plt.subplots()
plt.bar(x, height=values_very_n)
plt.xticks(x, names_very_n)
plt.xticks(fontsize=25, rotation=90)
# from matplotlib.ticker import FuncFormatter
# formatter = FuncFormatter(millions)
#
# ax.yaxis.set_major_formatter(formatter)

plt.yticks(fontsize=20)

plt.savefig('Distribution.pdf', bbox_inches='tight')
# plt.show()