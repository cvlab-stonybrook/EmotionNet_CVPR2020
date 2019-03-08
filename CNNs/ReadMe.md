# Instructions

This is a template following other templates files to train models

To train a model, you need to following

to use the code you need:

python 3
pytorch 1.0/0.4
tqdm

the main entrance code is in CNNs/mains. basically all of the files in this directory are similar to each other. You can modify your own following CNNs/mains/main_mclss_cross_entropy_v2.py



1. a script defining the parameters (saved in scripts) if you're training a mutliple-label classification problem, please refer to Adobe_Selected690_MultiClass_CrossEntropy_tag_based_config.json
2. The main entrance code is defined in mains. Please refer to mains/main_mclss_cross_entropy_v2.py for reference
3. You need to prepare your own pkl file with the format [relative_path, [labels]], you can to write your own data loader referring CNNs/datasets/multilabel.py

to execute the file you can jump into the mains directory do something like:
python main_mclass_cross_entropy_v2.py --config_file ../scripts/[your config file]
