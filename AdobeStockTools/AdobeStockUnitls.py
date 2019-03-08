# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 12/Feb/2019 16:57

from PyUtils.file_utils import get_stem

def get_image_cid_from_url(image_url, location=2):
    image_name = get_stem(image_url)
    image_cid = image_name.split('_')[location]
    return image_cid