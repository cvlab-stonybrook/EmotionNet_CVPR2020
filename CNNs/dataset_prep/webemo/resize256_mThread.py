# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 22/Oct/2018 18:18

import os, sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/Emotion3D')
sys.path.append(project_root)
import urllib.request
import progressbar
# import os
import tqdm
from multiprocessing import Pool
from PIL import Image
# import shutil

#
# urllib.request.urlretrieve("https://as2.ftcdn.net/jpg/01/61/57/81/1000_F_161578114_DBBHSHwBLXhMLdR0NRxTmthERwK3IW9L.jpg", "local-filename.jpg")
# ('local-filename.jpg', <http.client.HTTPMessage object at 0x7f829ea9dda0>)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """


    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)







def download_wrapper(s_input):
    """Wrapper for parallel processing purposes."""
    # cid, image_url, image_savename = s_input
    s_src_file, s_dst_file = s_input
    try:
        src_image = pil_loader(s_src_file)
        dst_image = resize(src_image, 256)
        dst_image.save(s_dst_file)
        return True, 'Success!'
    except:
    # status = tuple([cid, downloaded, log])
        return False, s_src_file

if __name__ == '__main__':

    from PyUtils.pickle_utils import loadpickle
    from PyUtils.file_utils import get_dir
    import glob

    # annotation_list = []
    max_idx = 0
    subdirectories = set()
    src_path = '/mnt/data/zwei/datasets/webemo-tars/webemo-images'
    dst_path = '/mnt/data/zwei/datasets/webemo-tars/webemo-images-256'

    filenames = glob.glob(os.path.join(src_path, '**/*.jpg'), recursive=True)

    operating_files = []
    for s_filename in tqdm.tqdm(filenames):
        s_path_parts = s_filename.split(os.sep)
        s_rel_path = s_path_parts[-3:]
        s_dst_path = os.path.join(dst_path, *s_rel_path)
        subdirectories.add(os.path.dirname(s_dst_path))
        operating_files.append((s_filename, s_dst_path))

    print("Create Directories")
    subdirectories = list(subdirectories)
    for s_subdir in tqdm.tqdm(subdirectories, desc="Creating Directories"):
        get_dir(os.path.join(dst_path, s_subdir))

    from multiprocessing import Pool

    pool = Pool(processes=64)
    for s_status in tqdm.tqdm(pool.imap_unordered(download_wrapper, operating_files), total=len(operating_files),
                              desc="Copying"):
        # if len(s_status[0]) > 0:
        #     CID_EmotionVectors[s_status[1]] = s_status[0]
        if not s_status[0]:
            print("Cannot copy {}".format(s_status[1]))

    print('DB')






