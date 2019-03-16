# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 15/Mar/2019 16:04
import torch.nn as nn
import torch
import torch.nn.functional as F


class MclassCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MclassCrossEntropyLoss, self).__init__()

    def forward(self, predicts, labels):
        log_softmax_output = F.log_softmax(predicts, dim=1)
        loss_cls = - torch.sum(log_softmax_output * labels) / predicts.shape[0]
        return loss_cls