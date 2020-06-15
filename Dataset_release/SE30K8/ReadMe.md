# SE30K8

A selected set of stock images annotated with 8 emotions 

``` text
0	anger
1	disgust
2	fear
3	happiness
4	neutral
5	sadness
6	surprise(negative)
7	surprise(positive)
```

## Annotation Protocal:

5 annotators from Mturk, multiple-label annotation is supported. i.e., everytime the user is allowed to annotated
 more than 1 emotion
 

## Data format

The data is saved in `Dataset_release/SE30K8/annotations/mturk_annotations_240.pkl.keep`. which is a dictionary of
 image annotations:

`access_key`: image_cid, integer format

`image_emotion`: (5) annotations provided by the mturks

`image_url`: downloadable url link to download the image

`work_ids`: mturk ids

`tags`: the tags that come with the stock image

`emotion_tags`: the emotion-related tags (690) that come with the stock image

`keys`: the keywords used to get this image in an internal search engine

`re_path`: NOT useful now. the relative path in the full EmotionStock Dataset


## Accessing the images:

you have two options: 



1. Download from Stony Brook Computer Vision Lab (Updated soon)

2. (For preview purpose only) Use URLs provided in the [annotation file](annotations/mturk_annotations_240.pkl.keep)

## More information

As each image comes with raw tags and emotion-related tags, you may need the tag2id or emotiontag2id mappings to map
 the (emotion) tag names into integer categories. For these information, please check out the following files in the
  StockEmotion dataset.
 
* tag2idx.pkl: the tag2idx, idx2tag conversion (convert the raw tags to cids)

* etag2idx.pkl: convert emotion tags to idx (690)

* etag2tagidx.pkl: convert emotion tags to their raw-tag ids
 
For more information, check the [StockEmotion Dataset](../StockEmotion/ReadMe.md)
