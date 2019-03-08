# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 09/Oct/2018 10:09


import json
# import munch
from argparse import Namespace

def update_config(raw_args, config_file):

    config_dict = json.load(open(config_file))
    raw_arg_dict = vars(raw_args)
    # TODO: compare raw_arg_dict and config dict
    raise NotImplementedError
    # config = munch.munchify(config_dict)
    # return config


def parse_config(config_file):
    config_dict = json.load(open(config_file))
    config = Namespace(**config_dict)
    # config = munch.munchify(config_dict)
    return config

if __name__ == '__main__':
    config_file = '/home/zwei/Dev/AttributeNet3/CNNs/scripts/Desk_Selected742_MultiClass_CrossEntropy_config.json'
    config = parse_config(config_file)