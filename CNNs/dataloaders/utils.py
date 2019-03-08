# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 16/Oct/2018 13:50

from torch.utils.data.dataloader import default_collate

def none_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return default_collate(batch)


if __name__ == '__main__':
    pass