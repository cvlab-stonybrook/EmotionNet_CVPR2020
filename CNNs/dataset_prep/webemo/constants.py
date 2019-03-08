# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 22/Oct/2018 22:24
import os
cate_25_dir = '/home/zwei/datasets/emotion_datasets/webemo/sampled_images/25-category/'
sentiment_type_2 = ['Negative', 'Positive']
emotion_type_6 = sorted(['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])
emotion_type_25 = [d for d in os.listdir(cate_25_dir) if os.path.isdir(os.path.join(cate_25_dir, d))]
emotion_type_25 = sorted(emotion_type_25)

labels=[('affection', 'love', '+'),
    ('cheerfullness', 'joy', '+'),
    ('confusion', 'confusion', '-'),
    ('contentment', 'joy', '+'),
    ('disappointment', 'sadness', '-'),
    ('disgust', 'anger', '-'),
    ('enthrallment', 'joy', '+'),
    ('envy', 'anger', '-'),
    ('exasperation', 'anger', '-'),
    ('gratitude', 'love', '+'),
    ('horror', 'fear', '-'),
    ('irritabilty', 'anger', '-'),
    ('lust', 'love', '+'),
    ('neglect', 'sadness', '-'),
    ('nervousness', 'fear', '-'),
    ('optimism', 'joy', '+'),
    ('pride', 'joy', '+'),
    ('rage', 'anger', '-'),
    ('relief', 'joy', '+'),
    ('sadness', 'sadness', '-'),
    ('shame', 'sadness', '-'),
    ('suffering', 'sadness', '-'),
    ('surprise', 'surprise', '+'),
    ('sympathy', 'sadness', '-'),
    ('zest', 'joy', '+')]