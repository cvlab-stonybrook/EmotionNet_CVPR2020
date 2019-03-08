# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 29/Oct/2018 10:56
from PyUtils.pickle_utils import loadpickle, save2pickle
from JM.utils import get_lexicons_compiled_dict

tag_frequencies = loadpickle('/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/train-word-frequencies.pkl')
lexicon_dict = get_lexicons_compiled_dict()

sentiments = {0: 'neutral', 1: 'negative', 2: 'positive'}
subjectivity = {0: 'weak', 1: 'strong'}

for idx, s_tag in enumerate(tag_frequencies):
    if s_tag in lexicon_dict:
        s_lexicon = lexicon_dict[s_tag]
        x_sentiment = s_lexicon['sentiment']
        x_subjectivity = s_lexicon['subjectivity']
        if x_sentiment == '':
            x_sentiment = 'None'
        if x_subjectivity == '':
            x_subjectivity = 'None'

        # if x_sentiment != '' and x_subjectivity!='':
        print("{} | {}({}): \tSentiment {}\tSubjectivity: {}".format(idx, s_tag, tag_frequencies[s_tag], x_sentiment, x_subjectivity))

        # while True:
        #     if x_sentiment != '' and x_subjectivity!='':
        #         break
        #
        #     if x_sentiment == '':
        #         input_sentiment = input("Input the sentiment (0: 'neutral', 1: 'negative', 2: 'positive'):\n")
        #         x_sentiment = sentiments[int(input_sentiment)]
        #
        #     if x_subjectivity == '':
        #         input_subjectivity = input("Input the subjectivity (0: 'weak', 1: 'strong'):\n")
        #         x_subjectivity = subjectivity[int(input_subjectivity)]
        #
        #     print("Updated {}: \tSentiment {}\tSubjectivity: {}".format(s_tag, x_sentiment, x_subjectivity))
        #     isConfirm = input("Confirm? (y/n)\n")
        #     if isConfirm.lower() == 'y':
        #         s_lexicon['sentiment'] = x_sentiment
        #         s_lexicon['subjectivity'] = x_subjectivity
        #         s_lexicon['updated']=True
        #         lexicon_dict[s_tag] = s_lexicon
        #         break
        #     else:
        #          x_sentiment = '', x_subjectivity != ''

    else:
        if tag_frequencies[s_tag] >100:
            print("{} | {} ({}) not found in lexicon_dict".format(idx, s_tag, tag_frequencies[s_tag]))
            # s_lexicon = {'word': s_tag, 'emotion': '', 'color':'', 'orientation':'', 'sentiment': '', 'subjectivity': '', 'source':'zwei'}
            # x_sentiment = s_lexicon['sentiment']
            # x_subjectivity = s_lexicon['subjectivity']
            # print("Creating: word {}: \tSentiment {}\tSubjectivity: {}".format(s_tag, x_sentiment, x_subjectivity))
            #
            # while True:
            #     if x_sentiment != '' and x_subjectivity!='':
            #         break
            #
            #     if x_sentiment == '':
            #         input_sentiment = input("Input the sentiment (0: 'neutral', 1: 'negative', 2: 'positive'):\n")
            #         x_sentiment = sentiments[int(input_sentiment)]
            #
            #     if x_subjectivity == '':
            #         input_subjectivity = input("Input the subjectivity (0: 'weak', 1: 'strong'):\n")
            #         x_subjectivity = sentiments[int(input_subjectivity)]
            #
            #     print("Updated {}: \tSentiment {}\tSubjectivity: {}".format(s_tag, x_sentiment, x_subjectivity))
            #     isConfirm = input("Confirm?\n")
            #     if isConfirm.lower() == 'y':
            #         s_lexicon['sentiment'] = x_sentiment
            #         s_lexicon['subjectivity'] = x_subjectivity
            #         lexicon_dict[s_tag] = s_lexicon
            #         break
            #     else:
            #          x_sentiment = '', x_subjectivity != ''

print("Done, Saving")

# save2pickle('lexicon_updated.pkl', lexicon_dict)