# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 22/Oct/2018 22:24
import os
sentiment_type_2 = ['Negative', 'Positive']

emotion_type_6 = sorted(['anger', 'fear', 'joy', 'love', 'sadness', 'surprise'])


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