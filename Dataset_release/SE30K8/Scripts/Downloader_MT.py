# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 12/Feb/2019 15:57


import os, sys

# project_root = os.path.join(os.path.expanduser('~'), 'Dev/Emotion3D')
# sys.path.append(project_root)
import urllib.request
import tqdm
from multiprocessing import Pool
from PIL import Image
# from PyUtils.json_utils import load_json_list

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




def download_image_resize(image_url, image_savename, MAX_ATTEMPTS=6):
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




def download_image(image_url, image_savename, MAX_ATTEMPTS=6):
    attempts = 0
    if os.path.exists(image_savename):
        return True, 'Exists'
    while attempts < MAX_ATTEMPTS:
        try:
            # tmp_save_place = os.path.join('/dev/shm', os.path.basename(image_savename))
            urllib.request.urlretrieve(image_url, image_savename)

            # src_image = pil_loader(tmp_save_place)
            # dst_image = resize(src_image, 256)
            # dst_image.save(image_savename)
            # os.remove(tmp_save_place)
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

def replace_240(url_1k):
    url_path = os.path.dirname(url_1k)
    url_image_name = os.path.basename(url_1k)
    url_image_parts = url_image_name.split('_')
    new_url_image_name = '_'.join(['240'] + url_image_parts[1:])
    return os.path.join(url_path, new_url_image_name)

if __name__ == '__main__':

    import os, sys
    from PyUtils.pickle_utils import loadpickle

    import argparse
    import glob
    input_file = '/home/zwei/Dev/PastProjects/AttributeNet3/Dataset_release/SE30K8/annotations/mturk_annotations_240.pkl.keep'
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', '-b', default=0, type=int, help="Start Idx")
    parser.add_argument('--end_idx', '-e', default=None, type=int)

    parser.add_argument('--input_file', '-i', default=input_file, type=str, help='Annotation File directory')
    parser.add_argument('--output_directory', '-t', default='SE30K8', type=str, help='output filename')
    parser.add_argument('--nworkers', '-n', default=8, type=int, help="Number of workers")
    args = parser.parse_args()

    annotations = loadpickle(args.input_file)
    # assert 'image_url' in annotations[0], "url is not an attributes of json file"
    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(annotations)
    if end_idx>len(annotations):
        end_idx = len(annotations)

    # selected_annotations = annotations[start_idx:end_idx]
    selected_annotations = annotations
    print("Downloading {0} File Records".format(len(selected_annotations)))
    MAX_attempt = 3

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    log_file = os.path.join(args.output_directory,'errlog-{0}-{1}-256.txt'.format(start_idx, end_idx))
    download_tuples = []

    every_n_image = 10000

    print("Processing candidates")

    for s_idx, (s_cid, s_annotation) in enumerate(tqdm.tqdm(selected_annotations.items())):

            # s_cid = s_annotation['cid']
            s_sub_directory = os.path.join(args.output_directory, '{:06d}'.format(s_idx // every_n_image))
            if not os.path.exists(s_sub_directory):
                os.makedirs(s_sub_directory)


            s_url =s_annotation['image_url']
            image_format = os.path.splitext(os.path.basename(s_url))[1]

            image_full_name = os.path.join(s_sub_directory,   '{0}{1}'.format(s_cid, image_format))
            s_tuple = (s_cid, s_url, image_full_name)
            download_tuples.append(s_tuple)

    print("DownLoading {0} Items".format(len(download_tuples)))
    num_jobs = args.nworkers

    pool = Pool(processes=num_jobs)
    with open(log_file, 'w') as of_:
        for s_status in tqdm.tqdm(pool.imap_unordered(download_wrapper, download_tuples), total=len(download_tuples)):
            if s_status[1]:
                continue
            else:
                of_.write('{:d}\n'.format(s_status[0]))

    print("Done")
