# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 22/Jan/2019 10:03


import  csv
import json
import os, sys

project_name = 'AttributeNet3'
project_root = os.path.join(os.path.expanduser('~'), 'Dev', project_name)
sys.path.append(project_root)

LEXICON_FILE = os.path.join(project_root,'EmotionLexicons/lexicons_compiled.csv')
CATEGORIES_FILE = os.path.join(project_root,'EmotionLexicons/categories.json')

emotion_lexicon_vocabulary = {}
# words = []
# categories = {}
# category_headers = []

with open(LEXICON_FILE, 'r') as f:
    rows = csv.reader(f, delimiter=',')
    emotion_lexicon_headers = next(rows, None)# remove header
    for row in rows:
        entry = {}
        for i, h in enumerate(emotion_lexicon_headers):
            entry[h] = row[i]
        if entry['word'] not in emotion_lexicon_vocabulary:
            emotion_lexicon_vocabulary[entry['word']] = entry
        else:
            print("{} Already in dict".format(entry['word']))
    # emotion_lexicon_words = [v['word'] for v in emotion_lexicon_vocabulary]

# Read categories
with open(CATEGORIES_FILE) as f:
    emotion_lexicon_categories = json.load(f)
    emotion_lexicon_category_headers = emotion_lexicon_categories.keys()


# print("DEB")