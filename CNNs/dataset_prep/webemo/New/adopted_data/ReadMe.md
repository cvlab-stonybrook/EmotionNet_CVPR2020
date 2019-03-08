# Files:

1. test/train-CIDs-fullinfo:

 directly copied from outside files. there are overlaps between train/val (fatal errors from Rameswar)

 data are orginzed in lists. for one image there can be multiple entries, representing multiple labels of a possible image


2. test/train-CIDs-fullinfo-non-repeat

there is no longer overlaps in train/val, train: 178707; val: 37507
data are orginized in dict. Each entry is an image, one image may have multuple labels

the entries of  saved data:
0. CID, 1, list(tags), 2, str(title) 3. category 25, 4. categroy 6, 5: category 2, 6, relative path

the name2idx of category6. category25 and category 2 can be found in constants file:
/home/zwei/Dev/AttributeNet/CNNs/dataset_prep/webemo/constants.py