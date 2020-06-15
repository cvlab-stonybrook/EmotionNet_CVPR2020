# Learning Visual Emotion Representations From Web Data	



By Zijun Wei, Jianming Zhang, Zhe Lin, Joon-Young Lee, Niranjan Balasubramanian, Minh Hoai, Dimitris Samaras


**in progress**


## Introduction:

This repo provides code structure for the experiments in "Learning Visual Emotion Representations From Web Data".

## Setup

### Requirements

```text
python 3
pytorch 1.0/0.4
tqdm
```

### Code structure & how to train or evaluate a model:

the main entrance code is in `CNNs/mains`. basically all of the files in this directory are similar to each other. You can modify your own following 
`CNNs/mains/main_mclss_cross_entropy_v2.py`

#### train a model

1. a script defining the parameters (saved in `scripts`) if you're training a mutliple-label classification problem, please refer to `Adobe_Selected690_MultiClass_CrossEntropy_tag_based_config.json`
2. The main entrance code is defined in mains. Please refer to mains/main_mclss_cross_entropy_v2.py for reference
3. You need to prepare your own pkl file with the format `[relative_path, [labels]]`, you can to write your own data loader referring CNNs/datasets/multilabel.py

to execute the file you can jump into the mains directory do something like:
```shell script
python main_mclass_cross_entropy_v2.py --config_file ../scripts/[your config file]
```

#### evaluate a model:

the same as training a model, modify the config file to set the corresponding parameters

## Trained Models:

pretrained model with softmax + embedding distance loss (on Google Drive.):

[model_2branch](https://drive.google.com/file/d/1jjVOpard4dhSb9t_9TjPly1p3ijs2VqN/view?usp=sharing)


fine-tuned models (with all layers fixed except FC) on benchmark datasets:

[UnbiasedEmotionModel](https://drive.google.com/file/d/1gSLmDsL-k97jCecT39-TYGSUOAcKLXFZ/view?usp=sharing)



## Datasets:



### StockEmotion Full Dataset


### SE30K8:

See `Dataset_release/SE30K8/ReadMe.md` for details

### StockEmotion

See `Dataset_release/StockEmotion/ReadMe.md` for details

## Citations:

please cite:
```text
@inproceedings{wei2020learning,
  title={Learning Visual Emotion Representations From Web Data},
  author={Wei, Zijun and Zhang, Jianming and Lin, Zhe and Lee, Joon-Young and Balasubramanian, Niranjan and Hoai, Minh and Samaras, Dimitris},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13106--13115},
  year={2020}
}```