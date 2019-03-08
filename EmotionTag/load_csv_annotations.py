# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 08/Feb/2019 17:48

import os

def get_tag_list_from_csv(csv_file):
    emotion_tag_list = []
    with open(csv_file, 'r') as of_:
        lines = of_.readlines()
        for s_line in lines:
            s_line_parts = s_line.strip().split(',')
            if len(s_line_parts[2]) > 0:
                if int(s_line_parts[2]) == 1:
                    emotion_tag_list.append(s_line_parts[0])
                else:
                    print("Unrecognizeable bit: {}".format(s_line.strip()))

    return emotion_tag_list

def get_full_tag_list(csv_file):
    emotion_tag_list = []
    emotion_frequencies = {}
    with open(csv_file, 'r') as of_:
        lines = of_.readlines()
        for s_line in lines:
            s_line_parts = s_line.strip().split(',')

            emotion_tag_list.append(s_line_parts[0])
            emotion_frequencies[s_line_parts[0]] = int(s_line_parts[1])
    return emotion_tag_list,emotion_frequencies


def load_verified(csv_file=None):
    if csv_file is None:
        csv_file = '/home/zwei/Dev/AttributeNet3/EmotionTag/emotion_annotations_csv/Verify_20190210.csv'
    emotion_transfer_dict = {}
    emotion_complete_list = []
    emotion_countable_list = []
    with open(csv_file, 'r') as of_:
        lines = of_.readlines()
        for s_line in lines[1:]:
            s_line_parts = s_line.strip().split(',')
            if len(s_line_parts[-3])>0 and int(s_line_parts[-3])== 1:
                emotion_complete_list.append(s_line_parts[0])
                if len(s_line_parts[-1]) < 1:
                    emotion_countable_list.append(s_line_parts[0])
                else:
                    emotion_transfer_dict[s_line_parts[0]] = s_line_parts[-1]

    return emotion_countable_list, emotion_transfer_dict, emotion_complete_list

excluded_emotions = {'perplexity', 'offend', 'disappoint', 'trepidation', 'embarrass', 'exasperation', 'unbearable', 'insulting', 'humiliate', 'condemn', 'disappointing'}

if __name__ == '__main__':
    verfied_emotion_file = '/home/zwei/Dev/AttributeNet3/EmotionTag/emotion_annotations_csv/Verify_20190210.csv'
    verfied_emotion_list, transer_dict, complete_list = load_verified(verfied_emotion_file)
    print("DEB")
    # from EmotionLexicons import constants
    # emotion_lexicon = constants.emotion_lexicon_vocabulary
    # JM_file = '/home/zwei/Dev/AttributeNet3/EmotionTag/emotion_annotations_csv/Jianming20190208.csv'
    # ZJ_file = '/home/zwei/Dev/AttributeNet3/EmotionTag/emotion_annotations_csv/ZJ20190208.csv'
    #
    #
    # full_emotion_taglist, emotion_frequencies = get_full_tag_list(ZJ_file)
    #
    # JM_list = get_tag_list_from_csv(JM_file)
    # ZJ_list =get_tag_list_from_csv(ZJ_file)
    #
    # ZJ_set = set(ZJ_list)
    # JM_set = set(JM_list)
    # assert len(ZJ_list) == len(ZJ_set) and len(JM_list) == len(JM_set), "The length of list or set should be the same!"
    # # the common list:
    # common_set = JM_set.intersection(ZJ_set)
    # # the tags annotated by ZJ but not by JM
    #
    # tag_JM_only = []
    # for s_tag in JM_list:
    #     if s_tag not in ZJ_set:
    #         tag_JM_only.append(s_tag)
    #
    #
    # # the tags annotated by JM but not by ZJ
    # tag_ZJ_only = []
    # for s_tag in ZJ_list:
    #     if s_tag not in JM_set:
    #         tag_ZJ_only.append(s_tag)
    #
    #
    #
    # tag_ZJ_only_set = set(tag_ZJ_only)
    # tag_JM_only_set = set(tag_JM_only)
    #
    #
    # tag_properties = {}
    # for s_tag in full_emotion_taglist:
    #     s_item = {}
    #     if s_tag in emotion_lexicon:
    #         s_tag_lexicon_props = emotion_lexicon[s_tag]
    #         if len(s_tag_lexicon_props['emotion'])>0 and s_tag_lexicon_props['subjectivity']=='strong':
    #             s_item['NRC-strong'] = 1
    #         else:
    #             s_item['NRC-strong'] = 0
    #
    #     else:
    #             s_item['NRC-strong'] = 0
    #     if s_tag in common_set:
    #         s_item['common'] = 1
    #     else:
    #         s_item['common'] = 0
    #
    #     if s_tag in tag_JM_only_set:
    #         s_item['JM'] = 1
    #     else:
    #         s_item['JM'] = 0
    #
    #     if s_tag in tag_ZJ_only_set:
    #         s_item['ZJ'] = 1
    #     else:
    #         s_item['ZJ'] = 0
    #
    #     assert s_item['ZJ'] + s_item['JM'] + s_item['common'] <= 1
    #     if s_item['ZJ'] + s_item['JM'] + s_item['common'] + s_item['NRC-strong']>=1:
    #         tag_properties[s_tag] = s_item
    #
    # with open('labelling_results_20190209.csv', 'w') as of_:
    #     of_.write('word,NRC-strong,common,JM_only,ZJ_only,frequencies\n')
    #     for s_tag in tag_properties:
    #         of_.write('{},{},{},{},{},{}\n'.format(s_tag, 1 if tag_properties[s_tag]['NRC-strong'] == 1 else "",
    #                                                1 if tag_properties[s_tag]['common'] == 1 else "",
    #                                                1 if tag_properties[s_tag]['JM']==1 else "",
    #                                                1 if tag_properties[s_tag]['ZJ']==1 else "",
    #                                                emotion_frequencies[s_tag]))
    #
    #
    #
    # print("DB")