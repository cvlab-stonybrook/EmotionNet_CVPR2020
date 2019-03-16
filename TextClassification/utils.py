# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 02/Mar/2019 19:25

import pickle
import torch
from PyUtils.file_utils import get_file_dir





def save_model(model, params):
    file_path = get_file_dir(params['save_path'])

    saved_items = {"model": model}
    torch.save(saved_items, file_path)
    print("A model is saved successfully as {}!".format(file_path))


# def load_model(params):
#     path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
#
#     try:
#         model = pickle.load(open(path, "rb"))
#         print(f"Model in {path} loaded successfully!")
#
#         return model
#     except:
#         print(f"No available model such as {path}.")
#         exit()
