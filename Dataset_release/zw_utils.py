#
# This file is part of the AttributeNet3 project.
#
# @author Zijun Wei <zwei@adobe.com>
# @copyright (c) Adobe Inc.
# 2020-Jun-15.
# 08: 15
# All Rights Reserved
#

import os

def replace_240(url_1k):
    url_path = os.path.dirname(url_1k)
    url_image_name = os.path.basename(url_1k)
    url_image_parts = url_image_name.split('_')
    new_url_image_name = '_'.join(['240'] + url_image_parts[1:])
    return os.path.join(url_path, new_url_image_name)