# Results:

## UnBiasedEmo:

6 classes

Softmax training

1. feature-extractor: fix top layers, learning rate 0.1 [10, 20 ,30], Best: .59
    file: /home/zwei/Dev/AttributeNet/ckpts/UnbiasedEmo-ImageNet-feature-extractor-6/main_deepemotion-feature-extractor-20181010223147

    Perhaps this can be better using more epochs...

2. fine-tune: using all layers but with smaller learning rate,
    file: /home/zwei/Dev/AttributeNet/CNNs/scripts/UnbiasedEmo_ImageNet_finetune_6_config.json

    This is getting better, the best is around 0.69


3. fine-tune on top of the Attribute1000K
    file: /home/zwei/Dev/AttributeNet/CNNs/scripts/UnbiasedEmo_Attri1K_finetune_6_config.json
   The best is around 0.59-0.62



4. Do nothing, just cold start: 0.32