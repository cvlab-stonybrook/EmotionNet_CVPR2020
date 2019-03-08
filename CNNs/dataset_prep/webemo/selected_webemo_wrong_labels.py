# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 23/Oct/2018 18:16

wrong_PN_tags = [85372487, 102481969, 64424599, 82508425, 2792428, 57282041, 106657281, 36213062, 50192213, 73226708, 86236512,
                 70460486, 85634791, 87399836]


if __name__ == '__main__':
    import glob
    import os
    from PyUtils.file_utils import get_stem
    # wrong_PN_tags = set(wrong_PN_tags)
    file_list = glob.glob(os.path.join('/home/zwei/datasets/emotion_datasets/webemo/selected_annotated', '*.jpg'))
    CIDs = []
    for s_file in file_list:
        s_ID = get_stem(os.path.basename(s_file)).split('-')[-1]
        CIDs.append(int(s_ID))

    CIDs = set(CIDs)
    assert len(CIDs) == 100
    for s_id in wrong_PN_tags:
        assert s_id in CIDs, '{} Not Found'.format(s_id)

    print("Done")