# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 29/Nov/2018 09:07
import os
import shutil



def mThreadCopy(inputs):
    if os.path.exists(inputs[1]):
        return True, "Exisit!"
    try:
        shutil.copyfile(inputs[0], inputs[1])
        return True, "Success"
    except:
        if not os.path.exists(inputs[0]):
            return False, "{} Not Exisit!".format(inputs[0])
        else:
            return False, '{} Other reasons!'.format(inputs[1])


if __name__ == '__main__':
    import tqdm
    from multiprocessing import Pool
    num_jobs = 100
    file2copy = ['src', 'dst']
    pool = Pool(processes=num_jobs)

    for s_status in tqdm.tqdm(pool.imap_unordered(mThreadCopy, file2copy), total=len(file2copy)):
        if not s_status[0]:
            print(s_status[1])