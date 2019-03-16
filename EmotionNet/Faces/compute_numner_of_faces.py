# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 13/Mar/2019 16:02

from PyUtils.pickle_utils import loadpickle_python2_compatible
from PyUtils.file_utils import get_stem
faces = loadpickle_python2_compatible('/home/zwei/datasets/stockimage_742/imageface_attributes/face_crop_index.pkl')

# for s_face in faces:
#     s_face_id = get_stem(s_face[0])[-3:]
#     if int(s_face_id) != 0:
#         print("{} is not the first face in image".format(s_face[0]))
# print("DB")


n_image_with_faces = 0
for s_face in faces:
    if len(s_face) >= 1:
        n_image_with_faces += 1


print("DB")