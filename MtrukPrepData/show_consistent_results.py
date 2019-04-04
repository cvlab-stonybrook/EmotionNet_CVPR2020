# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 07/Feb/2019 13:23
from PyUtils.pickle_utils import loadpickle
import argparse

def find_max_val_indict(input_dict):
    max_val = 0
    max_key = None
    for s_key in input_dict:
        if input_dict[s_key]>max_val:
            max_val = input_dict[s_key]
            max_key = s_key
    return max_val, max_key


def all_agrees(emotion_list):
    return list(set(emotion_list))
    # if len(set(emotion_list)) == 1:
    #     return emotion_list[:1]
    # else:
    #     return None



def has_x_agrees(emotion_list, x):
    emotion_counts = {}
    for s_emotion in emotion_list:
        if s_emotion in emotion_counts:
            emotion_counts[s_emotion] += 1
        else:
            emotion_counts[s_emotion] = 1

    max_val, max_key = find_max_val_indict(emotion_counts)
    if max_val == x:
        return max_key, emotion_counts
    else:
        return None, emotion_counts




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--image_results_file', default='/home/zwei/Dev/AttributeNet3/MturkCollectedData/results_imagebased/results_imagebased.txt_v3.pkl',
                        help="Provide then name of the image reuslt file.")
    parser.add_argument('--agree', type=int, default=2, help='minimal number of agrees')
    args = parser.parse_args()

    image_results = loadpickle(args.image_results_file)
    emotion_categories = ['happiness', 'sadness', 'fear', 'disgust', 'anger', 'surprise(positive)', 'surprise(negative)', 'neutral']
    emotion_category_counts = {}
    for s_emotion in emotion_categories:
        emotion_category_counts[s_emotion] = 0

    counts = 0
    for s_image_cid in image_results:
        image_emotions = []

        # assert len(image_results[s_image_cid]['worker_ids']) == len(image_results[s_image_cid]['emotion-annotations']) == len(image_results[s_image_cid]['tag-annotations'])
        for s_id in range(len(image_results[s_image_cid]['worker_ids'])):
            image_emotions.extend(image_results[s_image_cid]['image_emotion'][s_id])
                    # print('{};\t{}'.format(image_results[s_image_cid]['worker_ids'][s_id], ', '.join(image_results[s_image_cid]['image_emotion'][s_id])))

        output_emotion, output_counts = has_x_agrees(image_emotions, args.agree)
        if output_emotion is not None:
            emotion_category_counts[output_emotion]+=1
            counts +=1
            print("{}\t{}\tmajor emotion:{}\tall emotions: {}".format(counts, image_results[s_image_cid]['image_url'], output_emotion, '; '.join('{}({})'.format(key, output_counts[key]) for key in output_counts)))


    total_counts = 0
    for s_emotion in emotion_category_counts:
        total_counts += emotion_category_counts[s_emotion]
        print("{}\t{}".format(s_emotion, emotion_category_counts[s_emotion]))
    print("total: {}".format(total_counts))
    print("DB")