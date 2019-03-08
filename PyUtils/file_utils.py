import os
import os.path as osp
import datetime

def get_dir(directory):
    """
    Creates the given directory if it does not exist.
    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_file_dir(file_path):
    directory = os.path.dirname(file_path)
    get_dir(directory)
    return file_path

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')[:-7].replace('-', '')

def get_image_list(images):
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
    return imlist


def get_subdir_imagelist(directory):
    image_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[1] == '.jpg':
                image_list.append(os.path.join(path, name))
    return image_list


def get_stem(file_path):
    basename = os.path.basename(file_path)
    stem = os.path.splitext(basename)[0]
    return stem