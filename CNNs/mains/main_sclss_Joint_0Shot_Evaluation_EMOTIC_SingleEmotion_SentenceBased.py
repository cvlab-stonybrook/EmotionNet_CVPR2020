# Copyright (c) 2018 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# TODO: this is modified from main_mclass_corss_entropy_v2.py
# Email: hzwzijun@gmail.com
# Created: 07/Oct/2018 11:09
import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/AttributeNet3')
sys.path.append(project_root)
from sklearn.metrics.pairwise import cosine_similarity
# import argparse
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import CNNs.models as models
import CNNs.utils.util as CNN_utils
from torch.optim import lr_scheduler
from CNNs.dataloaders.utils import none_collate
from PyUtils.file_utils import get_date_str, get_dir, get_stem
from PyUtils import log_utils
import CNNs.datasets as custom_datasets
from CNNs.utils.config import parse_config
from CNNs.models.resnet import load_state_dict
import torch.nn.functional as F
from TextClassification.model_DAN_2constraints import Text_Transformation

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_instance(module, name, args):
    return getattr(module, name)(args)


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Image CNN training from Configure Files")
    parser.add_argument('--config_file', required=True, help="This scripts only accepts parameters from Json files")
    input_args = parser.parse_args()

    config_file = input_args.config_file

    args = parse_config(config_file)
    if args.name is None:
        args.name = get_stem(config_file)

    torch.set_default_tensor_type('torch.FloatTensor')
    best_prec1 = 0

    args.script_name = get_stem(__file__)
    current_time_str = get_date_str()





    print_func = print


    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    if args.pretrained:
        print_func("=> using pre-trained model '{}'".format(args.arch))
        visual_model = models.__dict__[args.arch](pretrained=True, num_classes=args.num_classes)
    else:
        print_func("=> creating model '{}'".format(args.arch))
        visual_model = models.__dict__[args.arch](pretrained=False, num_classes=args.num_classes)



    if args.gpu is not None:
        visual_model = visual_model.cuda(args.gpu)
    elif args.distributed:
        visual_model.cuda()
        visual_model = torch.nn.parallel.DistributedDataParallel(visual_model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            visual_model.features = torch.nn.DataParallel(visual_model.features)
            visual_model.cuda()
        else:
            # model = torch.nn.DataParallel(model).cuda()
            visual_model = visual_model.cuda()


    from PyUtils.pickle_utils import loadpickle

    import numpy as np
    from PublicEmotionDatasets.Emotic.constants import emotion_explainations_words_690 as emotion_self_words

    from torchvision.datasets.folder import default_loader
    tag_wordvectors = loadpickle('/home/zwei/Dev/AttributeNet3/TextClassification/visualizations/Embeddings/FullVocab_BN_transformed_l2_regularization.pkl')
    tag_words = []
    tag_matrix = []
    label_words = []
    label_matrix = []
    from TextClassification.model_DAN_2constraints import CNN_Embed_v2 as CNN
    text_ckpt = torch.load('/home/zwei/Dev/AttributeNet3/TextClassification/models/model_feature_regularization.pth.tar')
    text_saved_model = text_ckpt['model']
    params = {

        "MAX_SENT_LEN": text_saved_model.MAX_SENT_LEN,
        "BATCH_SIZE": text_saved_model.BATCH_SIZE,
        "WORD_DIM": text_saved_model.WORD_DIM,
        "FILTER_NUM": text_saved_model.FILTER_NUM,
        "VOCAB_SIZE": text_saved_model.VOCAB_SIZE,
        "CLASS_SIZE": text_saved_model.CLASS_SIZE,
        "DROPOUT_PROB": 0.5,
    }

    text_generator = CNN(**params).cuda()

    text_generator.load_state_dict(text_saved_model.state_dict(), strict=True)
    embedding_tag2idx = text_ckpt['tag2idx']
    text_generator.eval()

    text_model = Text_Transformation(300, 300, 8)
    text_model = text_model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print_func("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # text_model.load_state_dict(checkpoint['text_state_dict'])
            load_state_dict(text_model, checkpoint['text_state_dict'])

            import collections
            if isinstance(checkpoint, collections.OrderedDict):
                load_state_dict(visual_model, checkpoint, exclude_layers=['fc.weight', 'fc.bias'])


            else:
                load_state_dict(visual_model, checkpoint['state_dict'], exclude_layers=['module.fc.weight', 'module.fc.bias'])
                print_func("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print_func("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        print_func("=> This script is for fine-tuning only, please double check '{}'".format(args.resume))
        print_func("Now using randomly initialized parameters!")

    cudnn.benchmark = True







    from torch.autograd import Variable


    emotic_emotion_explaintations = {}

    for x_key in emotion_self_words:
        x_words = emotion_self_words[x_key].split(',')
        x_feature = [embedding_tag2idx[x] for x in x_words] + \
                    [text_saved_model.VOCAB_SIZE+1]*(text_saved_model.MAX_SENT_LEN - len(x_words))
        x_feature = Variable(torch.LongTensor(x_feature).unsqueeze(0)).cuda()

        tag_matrix = text_generator(x_feature)
        _, tag_feature = text_model(tag_matrix)
        # tag_matrix = tag_matrix.squeeze(1)
        item = {}
        item ['pred'] = []
        item ['label'] = []
        item ['target_matrix'] = tag_feature.cpu().data.numpy()
        item ['description'] = x_words
        emotic_emotion_explaintations[x_key] = item

    val_list = loadpickle('/home/zwei/datasets/PublicEmotion/EMOTIC/z_data/test_image_based_single_person_only.pkl')
    image_directory = '/home/zwei/datasets/PublicEmotion/EMOTIC/images'
    from CNNs.datasets.multilabel import get_val_simple_transform
    val_transform = get_val_simple_transform()
    visual_model.eval()

    import tqdm
    for i, (input_image_file, target, _, _) in tqdm.tqdm(enumerate(val_list), desc="Evaluating Peace",
                                                         total=len(val_list)):
        # measure data loading time

        image_path = os.path.join(image_directory, input_image_file)
        input_image = default_loader(image_path)
        input_image = val_transform(input_image)

        if args.gpu is not None:
            input_image = input_image.cuda(args.gpu, non_blocking=True)
        input_image = input_image.unsqueeze(0).cuda()

        # target_idx = target.nonzero() [:,1]

        # compute output
        output, output_proj = visual_model(input_image)

        output_proj = output_proj.cpu().data.numpy()
        target_labels = set([x[0] for x in target.most_common()])

        for x_key in emotic_emotion_explaintations:

            dot_product_label = cosine_similarity(output_proj, emotic_emotion_explaintations[x_key]['target_matrix'])[0]
            pred_score = np.average(dot_product_label)
            emotic_emotion_explaintations[x_key]['pred'].append(pred_score)
            if x_key in target_labels:
                emotic_emotion_explaintations[x_key]['label'].append(1)
            else:
                emotic_emotion_explaintations[x_key]['label'].append(0)

    from sklearn.metrics import average_precision_score
    full_AP = []
    for x_key in emotic_emotion_explaintations:
        full_pred = np.array(emotic_emotion_explaintations[x_key]['pred'])
        full_label = np.array(emotic_emotion_explaintations[x_key]['label'])
        AP = average_precision_score(full_label, full_pred)
        if np.isnan(AP):
            print("{} is Nan".format(x_key))
            continue
        full_AP.append(AP)
        print("{}\t{:.4f}".format(x_key, AP * 100))
    AvgAP = np.mean(full_AP)
    print("Avg AP: {:.2f}".format(AvgAP * 100))

    # print("* {} Image: {} GT label: {}, predicted label: {}".format(i, input_image_file, idx2emotion[target], idx2label[output_label]))
        # print(" == closest tags: {}".format(', '.join(['{}({:.02f})'.format(idx2tag[x], dot_product_tag[x]) for x in out_tags])))
    # print("Accuracy {:.4f}".format(correct/total))








if __name__ == '__main__':
    main()

