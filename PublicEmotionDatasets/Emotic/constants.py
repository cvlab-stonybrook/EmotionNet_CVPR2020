# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO):
# Email: hzwzijun@gmail.com
# Created: 08/Mar/2019 09:19

from PyUtils.dict_utils import string_list2dict


emotion_list = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
                'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
                'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity',
                'Suffering', 'Surprise', 'Sympathy', 'Yearning']

idx2emorion, emorion2idx = string_list2dict(emotion_list)

emotion_explainations_orig = {
    'Peace': 'well being and relaxed; no worry; having positive thoughts or sensations; satisfied',
    'Affection': 'fond feelings; love; tenderness',
    'Esteem': 'feelings of favorable opinion or judgment; respect; admiration; gratefulness',
    'Anticipation': 'state of looking forward; hoping on or getting prepared for possible future events',
    'Engagement': 'paying attention to something; absorbed into something; curious; interested',
    'Confidence': 'feeling of being certain; conviction that an outcome will be favorable; encouraged; proud',
    'Happiness': 'feeling delighted; feeling enjoyment or amusement',
    'Pleasure': 'feeling of delight in the senses',
    'Excitement': 'feeling enthusiasm; stimulated; energetic',
    'Surprise': 'sudden discovery of something unexpected',
    'Sympathy': 'state of sharing others emotions, goals or troubles; supportive; compassionate',
    'Doubt/Confusion': 'difficulty to understand or decide; thinking  about different options',
    'Disconnection': 'feeling  not interested in the main event of the surrounding; indifferent; bored; distracted',
    'Fatigue': 'weariness; tiredness; sleepy',
    'Embarrassment': 'feeling ashamed or guilty',
    'Yearning': 'strong desire to have something; jealous; envious; lust',
    'Disapproval': 'feeling that something is wrong or reprehensible; contempt; hostile',
    'Aversion': 'feeling disgust, dislike, repulsion; feeling hate',
    'Annoyance': 'bothered by something or someone; irritated; impatient; frustrated',
    'Anger': 'intense displeasure or rage; furious; resentful',
    'Sensitivity': 'feeling of being physically or emotionally wounded; feeling delicate or vulnerable',
    'Sadness': 'feeling unhappy, sorrow, disappointed, or discouraged',
    'Disquietment': 'nervous; worried; upset; anxious; tense; pressured; alarmed',
    'Fear': 'feeling suspicious or afraid of danger, threat, evil or pain; horror',
    'Pain': 'physical suffering',
    'Suffering': 'psychological or emotional pain; distressed; anguished',
}


emotion_explainations_words_690 = {
    'Peace': 'relaxed,satisfied',
    'Affection': 'fond,love,tenderness',
    'Esteem': 'respect,admiration,grateful',
    'Anticipation': 'hope,prepare,prepared',
    'Engagement': 'attention,attend,curious,interested',
    'Confidence': 'certain,conviction,encourage,proud',
    'Happiness': 'delighted,enjoyment,amusement',
    'Pleasure': 'delight',
    'Excitement': 'enthusiasm,stimulation,energetic',
    'Surprise': 'sudden,unexpected',
    'Sympathy': 'sharing,supportive,compassionate',
    'Doubt/Confusion': 'doubt,confusion',
    'Disconnection': 'boredom,bored,distracted,boring,dislike',
    'Fatigue': 'weary,tiredness,sleepy',
    'Embarrassment': 'ashamed,guilty,shame',
    'Yearning': 'desire,jealous,envious,envy,lust',
    'Disapproval': 'contempt,hostile,disapprove',
    'Aversion': 'disgust,dislike,repulsion,hate',
    'Annoyance': 'bothered,irritated,impatient,frustrated',
    'Anger': 'displeasure,rage,furious,resentful',
    'Sensitivity': 'delicate,vulnerable,touching',
    'Sadness': 'unhappy,sorrow,disappointed,discouraged',
    'Disquietment': 'nervous,worried,upset,anxious,tense,pressured,alarmed',
    'Fear': 'suspicious,afraid,danger,threat,evil,pain,horror',
    'Pain': 'suffering',
    'Suffering': 'pain,distressed,anguish',
}


emotion_self_words = {
    'Peace': 'peace',
    'Affection': 'affection',
    'Esteem': 'esteem',
    'Anticipation': 'anticipation',
    'Engagement': 'engagement',
    'Confidence': 'confidence',
    'Happiness': 'happiness',
    'Pleasure': 'pleasure',
    'Excitement': 'excitement',
    'Surprise': 'surprise',
    'Sympathy': 'sympathy',
    'Doubt/Confusion': 'doubt,confusion',
    'Disconnection': 'boredom',
    'Fatigue': 'fatigue',
    'Embarrassment': 'embarrassment',
    'Yearning': 'yearning',
    'Disapproval': 'disapproval',
    'Aversion': 'aversion',
    'Annoyance': 'annoyance',
    'Anger': 'anger',
    'Sensitivity': 'sensitivity',
    'Sadness': 'sadness',
    'Disquietment': 'nervous',
    'Fear': 'fear',
    'Pain': 'pain',
    'Suffering': 'suffering',
}


emotion_full_words_690 = {
    'Peace': 'peace,relaxed,satisfied',
    'Affection': 'affection,fond,love,tenderness',
    'Esteem': 'esteem,respect,admiration,grateful',
    'Anticipation': 'anticipation,hope,prepare,prepared',
    'Engagement': 'engagement,attention,attend,curious,interested',
    'Confidence': 'confidence,certain,conviction,encourage,proud',
    'Happiness': 'happiness,delighted,enjoyment,amusement',
    'Pleasure': 'pleasure,delight',
    'Excitement': 'excitement,enthusiasm,stimulation,energetic',
    'Surprise': 'surprise,sudden,unexpected',
    'Sympathy': 'sympathy,supportive,compassionate',
    'Doubt/Confusion': 'doubt,confusion',
    'Disconnection': 'boredom,bored,distracted,boring,dislike',
    'Fatigue': 'fatigue,weary,tiredness,sleepy',
    'Embarrassment': 'embarrassment,ashamed,guilty,shame',
    'Yearning': 'yearning,desire,jealous,envious,envy,lust',
    'Disapproval': 'disapproval,contempt,hostile,disapprove',
    'Aversion': 'aversion,disgust,dislike,repulsion,hate',
    'Annoyance': 'annoyance,bothered,irritated,impatient,frustrated',
    'Anger': 'anger,displeasure,rage,furious,resentful',
    'Sensitivity': 'sensitivity,delicate,vulnerable,touching',
    'Sadness': 'sadness,unhappy,sorrow,disappointed,discouraged',
    'Disquietment': 'nervous,worried,upset,anxious,tense,pressured,alarmed',
    'Fear': 'fear,suspicious,afraid,danger,threat,evil,pain,horror',
    'Pain': 'pain,suffering',
    'Suffering': 'suffering,pain,distressed,anguish',
}


def isInDict(word, dictionary):
    if word in dictionary:
        return True
    else:
        return False

if __name__ == '__main__':
    from PyUtils.pickle_utils import loadpickle

    emotion_vocabulary = loadpickle(
        '/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/Emotion_vocabulary.pkl')
    emotion_keys = emotion_vocabulary['key2idx']
    full_tag2idx = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/visualizations/Embeddings/FullVocab_BN_transformed_l2_regularization.pkl')


    for s_key in emotion_self_words:
        s_list = emotion_self_words[s_key].split(',')
        for x in s_list:
            # if x in emotion_keys:
            #     pass
            # else:
            #     print("{}\t{} Not found in 690".format(s_key, x))
            if x in full_tag2idx:
                pass
            else:
                print("{}\t{} Not found in full_vocab".format(s_key, x))


    print("DB")


    # emotion_vocab = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/visualizations/Embeddings/Key690_BN_transformed_l2_regularization.pkl')
    # full_vocab = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/visualizations/Embeddings/FullVocab_BN_transformed_l2_regularization.pkl')
    #
    # x_emotion = emotion_vocab['abandoned']
    # y_emotion = full_vocab['abandoned']
    #
    # emotion_explainations_orig = {
    # 'Peace': 'well being and relaxed; no worry; having positive thoughts or sensations; satisfied',
    # 'Affection': 'fond feelings; love; tenderness',
    # 'Esteem': 'feelings of favorable opinion or judgment; respect; admiration; gratefulness',
    # 'Anticipation': 'state of looking forward; hoping on or getting prepared for possible future events',
    # 'Engagement': 'paying attention to something; absorbed into something; curious; interested',
    # 'Confidence': 'feeling of being certain; conviction that an outcome will be favorable; encouraged; proud',
    # 'Happiness': 'feeling delighted; feeling enjoyment or amusement',
    # 'Pleasure': 'feeling of delight in the senses',
    # 'Excitement': 'feeling enthusiasm; stimulated; energetic',
    # 'Surprise': 'sudden discovery of something unexpected',
    # 'Sympathy': 'state of sharing others emotions, goals or troubles; supportive; compassionate',
    # 'Doubt / Confusion': 'difficulty to understand or decide; thinking  about different options',
    # 'Disconnection': 'feeling  not interested in the main event of the surrounding; indifferent; bored; distracted',
    # 'Fatigue': 'weariness; tiredness; sleepy',
    # 'Embarrassment': 'feeling ashamed or guilty',
    # 'Yearning': 'strong desire to have something; jealous; envious; lust',
    # 'Disapproval': 'feeling that something is wrong or reprehensible; contempt; hostile',
    # 'Aversion': 'feeling disgust, dislike, repulsion; feeling hate',
    # 'Annoyance': 'bothered by something or someone; irritated; impatient; frustrated',
    # 'Anger': 'intense displeasure or rage; furious; resentful',
    # 'Sensitivity': 'feeling of being physically or emotionally wounded; feeling delicate or vulnerable',
    # 'Sadness': 'feeling unhappy, sorrow, disappointed, or discouraged',
    # 'Disquietment': 'nervous; worried; upset; anxious; tense; pressured; alarmed',
    # 'Fear': 'feeling suspicious or afraid of danger, threat, evil or pain; horror',
    # 'Pain': 'physical suffering',
    # 'Suffering': 'psychological or emotional pain; distressed; anguished',
    # }










