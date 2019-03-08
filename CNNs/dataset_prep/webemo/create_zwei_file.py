# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): based on train/test files provided by Rameswar, set it to my costumized directories
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 17:34


# TODO: this script is more or less one-time thing, no need to make it fancy!

import os, sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet')
sys.path.append(project_root)

from PyUtils.pickle_utils import save2pickle
from PyUtils.file_utils import get_dir
import shutil
import argparse


# file_split = 'train'  # candidate: train_sample (1:1), test
# src_file_path = '/home/zwei/datasets/emotion_datasets/webemo/adobe-stock/correct_index_files/{}-images-25-6-2-category.txt'.format(file_split)
src_file_path = ''
root_path = '/home/zwei/datasets/emotion_datasets/webemo/adobe-stock/images'
dest_path = '/'

parser = argparse.ArgumentParser(description='Manage Information')

parser.add_argument('-s', '--source', default=root_path,
                    help='src directory for retrieved files')
parser.add_argument('-t', '--target', default=dest_path,
                    help='target directory to save files')

parser.add_argument('-f', '--file', default=src_file_path, help='src_file')
parser.add_argument('-n', '--nworkers', default=24)
def main():

    # data_set_information = []
    args = parser.parse_args()

    src_file_path = args.file
    root_path = args.source
    dest_path = args.target
    n_copied = 0
    n_error = 0
    with open(src_file_path, 'r') as if_:
        for line in if_:
            contents = line.strip().split(' ')
            abs_image_path = contents[0]


            abs_image_path_parts = abs_image_path.split(os.sep)

            rel_image_path = os.path.join(*abs_image_path_parts[-3:])
            rel_image_dir = get_dir(os.path.join(dest_path, os.path.join(*abs_image_path_parts[-3:-1])))
            rel_image_name = abs_image_path_parts[-1]
            if os.path.exists(os.path.join(root_path, rel_image_path)):
                # data_set_information.append((rel_image_path, new_label))
                src_file = os.path.join(root_path, rel_image_path)
                dst_file = os.path.join(rel_image_dir, rel_image_name)
                if os.path.exists(dst_file):
                    print("{0} Exist! Skip!".format(dst_file))
                else:
                    print(" {0} Copying: {1} --> {2}".format(n_copied, src_file, dst_file))
                    shutil.copyfile(src_file, dst_file)
                    n_copied += 1
            else:
                print(" *   *  {:s} Not Exist".format(rel_image_path))
                n_error += 1

    print("Done Preparing, {0} Copied, {1} Error!".format(n_copied, n_error))
    # num_jobs = args.nworkers
    #
    # pool = Pool(processes=num_jobs)
    # for s_status in tqdm.tqdm(pool.imap_unordered(vector_map_wrapper, CIDs), total=len(CIDs)):
    #     if len(s_status[0]) > 0:
    #         CID_EmotionVectors[s_status[1]] = s_status[0]

if __name__ == '__main__':
    main()








