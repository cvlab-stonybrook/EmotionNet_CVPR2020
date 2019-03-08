# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Mar/2019 22:56


from PyUtils.pickle_utils import loadpickle, save2pickle
import tqdm, os
from collections import Counter
from PyUtils.json_utils import load_json_list
from PyUtils.file_utils import get_stem, get_dir, get_file_dir
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def crop_image(src_file, target_file, coordinate ):

    if os.path.exists(target_file):
        return True, 'Exists'
    else:
        try:

            src_image = pil_loader(src_file)
            dst_image = src_image.crop(coordinate)

            dst_image.save(target_file)
            return True, 'Success'
        except:

            return False, 'Fail'







dataset_directory = '/home/zwei/datasets/PublicEmotion/EMOTIC'

original_image_directory = os.path.join(dataset_directory, 'images')
target_image_directory = get_dir(os.path.join(dataset_directory, 'images_face_crop'))


data_split = 'train'
annotation_file = os.path.join(dataset_directory, 'z_data', '{}_person_based.pkl'.format(data_split))


raw_annotation_list = loadpickle(annotation_file)


for s_annotation in tqdm.tqdm(raw_annotation_list, desc="Processing data"):
    s_target_file = s_annotation[0]
    s_bbox = s_annotation[-1]
    s_original_file_path = os.path.join(original_image_directory, '{}.jpg'.format(s_target_file[:-8]))
    s_target_file_path = os.path.join(target_image_directory, s_target_file)
    get_file_dir(s_target_file_path)
    if os.path.exists(s_original_file_path):
        crop_image(s_original_file_path, s_target_file_path, s_bbox)
    else:
        print("Something wrong with {}".format(s_original_file_path))



