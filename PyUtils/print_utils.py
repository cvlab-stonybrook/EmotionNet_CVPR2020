# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 27/Oct/2018 15:09
import numpy as np


def print_pred(scores, names, breaker=' '):
    assert len(scores) == len(names)
    idx = np.argsort(scores)[::-1]
    buff = ',{}'.format(breaker).join(['{}({:.2f})'.format(names[idx[i]],  scores[idx[i]]) for i in range(len(names))])
    print(buff)