# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 12/Feb/2019 15:57


import os, sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)
import urllib.request
import tqdm
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




def download_image(image_url, image_savename, MAX_ATTEMPTS=6):
    attempts = 0
    if os.path.exists(image_savename):
        return True, 'Exists'
    while attempts < MAX_ATTEMPTS:
        try:
            tmp_save_place = os.path.join('/dev/shm', os.path.basename(image_savename))
            urllib.request.urlretrieve(image_url, tmp_save_place)

            src_image = pil_loader(tmp_save_place)
            dst_image = resize(src_image, 256)
            dst_image.save(image_savename)
            os.remove(tmp_save_place)
            return True, 'Success'
        except:
            attempts += 1

    return False, 'Fail'


def download_wrapper(s_input):
    """Wrapper for parallel processing purposes."""
    cid, image_url, image_savename = s_input
    if os.path.exists(image_savename):
        status = tuple([cid, True, 'Exists'])
        return status

    downloaded, log = download_image(image_url, image_savename)

    status = tuple([cid, downloaded, log])
    return status

if __name__ == '__main__':

    import os, sys
    # from joblib import Parallel
    # from joblib import delayed
    project_root = os.path.join(os.path.expanduser('~'), 'Dev/Emotion3D')
    sys.path.append(project_root)
    from PyUtils.pickle_utils import loadpickle
    from AdobeStockTools.AdobeStockUnitls import get_image_cid_from_url
    import argparse
    import glob
    input_file = '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v2/image_download_list.pkl'
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', '-b', default=0, type=int, help="Start Idx")
    parser.add_argument('--end_idx', '-e', default=None, type=int)

    parser.add_argument('--input_file', '-s', default=input_file, type=str, help='Annotation File directory')
    parser.add_argument('--output_directory', '-t', default='/home/zwei/datasets/stockimage_742/images-256', type=str, help='output filename')
    parser.add_argument('--nworkers', '-n', default=24, type=int, help="Number of workers")
    args = parser.parse_args()

    image_urls = loadpickle(args.input_file)
    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(image_urls)
    if end_idx>len(image_urls):
        end_idx = len(image_urls)

    selected_urls = image_urls[start_idx:end_idx]
    print("Downloading {0} images".format(len(selected_urls)))
    MAX_attempt = 3

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    log_file = os.path.join(args.output_directory,'log-{0}-{1}-256.txt'.format(start_idx, end_idx))
    download_tuples = []


    print("Processing candidates")

    for s_idx, s_image_url in enumerate(tqdm.tqdm(selected_urls)):

            s_cid = get_image_cid_from_url(s_image_url[1])
            s_sub_directory = os.path.join(args.output_directory, os.path.dirname(s_image_url[0]))
            if not os.path.exists(s_sub_directory):
                os.makedirs(s_sub_directory)


            s_url =s_image_url[1]

            image_full_path = os.path.join(args.output_directory, s_image_url[0])
            s_tuple = (s_cid, s_url, image_full_path)
            download_tuples.append(s_tuple)

    print("DownLoading {0} Items".format(len(download_tuples)))
    num_jobs = args.nworkers

    pool = Pool(processes=num_jobs)
    with open(log_file, 'w') as of_:
        for s_status in tqdm.tqdm(pool.imap_unordered(download_wrapper, download_tuples), total=len(download_tuples)):
            if s_status[1]:
                continue
            else:
                of_.write('{}\n'.format(int(s_status[0])))

    print("Done")



