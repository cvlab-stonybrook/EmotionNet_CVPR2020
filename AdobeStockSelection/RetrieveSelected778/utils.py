# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 17/Feb/2019 12:48

import os
def convert_imagename2imagerelpath(imagename, splitchar='_', location=0, directory_digits=4, imagesperdirectory=5000):
    image_idx = int(imagename.split(splitchar)[location])
    #TODO FIXME
    directory = '{:04d}'.format(image_idx//imagesperdirectory)
    rel_path = os.path.join(directory, imagename)
    return rel_path
