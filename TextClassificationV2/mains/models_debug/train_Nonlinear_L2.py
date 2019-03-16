# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): training only 1 loss: the classification loss only
# Email: hzwzijun@gmail.com
# Created: 02/Mar/2019 18:43

from TextClassificationV2.models.TextCNN import TextCNN_NLT as TextCNN
from TextClassificationV2.data_loader.AMT import read_AMT_complete_mtrain_mtest

import TextClassification.utils as utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
import numpy as np
import argparse
import copy
from PyUtils.pickle_utils import loadpickle
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import random
from torch.optim import lr_scheduler
import tqdm
from CNNs.utils.util import AverageMeter
from CNNs.utils.util import report_params
from argparse import Namespace
import shutil
from PyUtils.file_utils import get_file_dir
# FIXME: update this later!
def ifContainCoreWords(tag2idx):
    emotion690_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/Emotion_vocabulary.pkl')
    selected_emotion690 = emotion690_vocabulary['key2idx']

    all_found = True
    for x in selected_emotion690:
        if x not in tag2idx:
                print("{} Not Found".format(x))
                all_found = False
    if all_found:
        print("All the 690 words can be found in this dict")


def create_data_args(data_splits):
    args_data = Namespace()
    args_data.vocab = sorted(list(set([w for x_split in data_splits for sent in x_split for w in sent[0]])))
    args_data.classes = [0, 1, 2, 3, 4, 5, 6, 7] # FIXME: do this in a better way!
    args_data.tag2idx = {w: i for i, w in enumerate(args_data.vocab)}
    args_data.idx2tag = {i: w for i, w in enumerate(args_data.vocab)}
    return args_data


def main():

    parser = argparse.ArgumentParser(description="CNN Text Classification with Pytorch")
    parser.add_argument("--epoch", default=80, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", '-lr', default=1.0, type=float, help="learning rate")
    parser.add_argument("--lr_schedule", '-lrs', default='40,80', type=str, help="the schedule of learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--word_dim', default=300, type=int, help="word dimension")
    parser.add_argument('--filter_dim', default=300, type=int, help="intermediate feature size, same as the word-dim will make it much easier")
    parser.add_argument('--max_len', default=100, type=int, help="length of each sentence, will set to max length if not given")
    parser.add_argument('--batch_size', default=50, type=int, help="batch-size")
    parser.add_argument('--save_path', default='models_debug/default_run1.pth.tar')
    parser.add_argument('--static', default=True)
    parser.add_argument('--max_norm', default=3.0, type=float)
    parser.add_argument('--w2v_type', default='selftrained_shuffle', help='selftrained | googlenews')

    args_hyper = parser.parse_args()
    args_data = Namespace()
    args_model = Namespace()
    get_file_dir(args_hyper.save_path)

    train_set, val_set = read_AMT_complete_mtrain_mtest()

    args_data = create_data_args([train_set, val_set])

    ifContainCoreWords(args_data.tag2idx)

    args_model = Namespace(
        max_len=args_hyper.max_len or max([len(sent[0]) for sent in train_set + val_set]),
        word_dim=args_hyper.word_dim,
        filter_num=args_hyper.filter_dim,
        vocab_size=len(args_data.vocab),
        num_classes=len(args_data.classes),
        dropout_prob=0.5,
        static=args_hyper.static,
        in_channel=1
    )

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("EPOCH:", args_hyper.epoch)
    print("LEARNING_RATE:", args_hyper.learning_rate)

    print("VOCAB_SIZE:", args_model.vocab_size)
    print("MAX_SENT_SIZE:", args_model.max_len)
    print("WORD_DIM:", args_model.word_dim)
    print("FILTER_NUM:", args_model.filter_num)

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    best_model = train(train_set, val_set, args_data, args_model, args_hyper)

    #TODO: save everything to

    save_items = {'state_dict': best_model.state_dict(), 'args_data': args_data, 'args_model': args_model}
    save_checkpoint(save_items, is_best=False, filename='../ckpts/{}_Nonlinear_L2.pth.tar'.format(args_hyper.w2v_type))
    # torch.save(save_items, args_hyper.save_path)



class MultiLabelCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiLabelCrossEntropy, self).__init__()

    def forward(self, predicts, labels):
        log_softmax_output = F.log_softmax(predicts, dim=1)
        loss = - torch.sum(log_softmax_output * labels) / predicts.shape[0]
        return loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def multilabelHits(pred_index, multilabel):
    n_correct = sum([1 if y[p] > 0 else 0 for p, y in zip(pred_index, multilabel)])
    n_pred = len(pred_index)
    acc = n_correct * 1. / n_pred
    return acc, n_correct, n_pred


def multilabelTop1(pred_index, multilabel):
    largest_vals = [max(x) for x in multilabel]
    n_correct = sum([1 if y[p] == largest_vals[i] else 0 for i, (p, y) in enumerate(zip(pred_index, multilabel))])
    n_pred = len(pred_index)
    acc = n_correct*1. / n_pred
    return acc, n_correct, n_pred



def train(train_set, val_set, args_data, args_model, args_hyper):
    wordvec_type = args_hyper.w2v_type  # or selftrained or googlenews
    wordvec_file = '/home/zwei/Dev/AttributeNet3/TextClassification/pre_extract_w2v/params/' \
                   '{}_extracted_w2v_wordnet_synsets_py3.pl'.format(wordvec_type)

    print("Loading from {}".format(wordvec_type))
    pretrained_w2v = loadpickle(wordvec_file)

    #This is only creating a vocabulary that exists in th
    wv_matrix = []
    words_not_found = []
    for i in range(len(args_data.vocab)):
        word = args_data.idx2tag[i]
        if word in pretrained_w2v:
            wv_matrix.append(pretrained_w2v[word])
        else:
            words_not_found.append(word)
            # print("{} not found in dictrionary, will use random".format(word))
            wv_matrix.append(np.random.uniform(-0.01, 0.01, args_hyper.word_dim).astype("float32"))
    print("{} words were not found".format(len(words_not_found)))
    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, args_hyper.word_dim).astype("float32"))
    wv_matrix.append(np.zeros(args_hyper.word_dim).astype("float32"))
    wv_matrix = np.array(wv_matrix)

    model = TextCNN(args_model, init_wv=wv_matrix).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, args_hyper.learning_rate)


    scheduler = lr_scheduler.MultiStepLR(
        optimizer, [int(x) for x in args_hyper.lr_schedule.split(',')], gamma=0.1)
    max_dev_top1 = 0
    max_dev_hits = 0
    max_test_acc = 0
    best_model = None
    model.train()

    report_params(model)
    for e in range(args_hyper.epoch):
        train_set = shuffle(train_set)
        train_losses = AverageMeter()
        train_top1 = AverageMeter()
        train_hits = AverageMeter()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        model.train()

        for i in tqdm.tqdm(range(0, len(train_set), args_hyper.batch_size)):
            batch_range = min(args_hyper.batch_size, len(train_set) - i)

            # add random dropping words:
            batch_x = []

            for sent in train_set[i:i + batch_range]:
                x_sent = sent[0]
                drop_thre = 0.2
                x_collected_words = []
                for x_word in x_sent:
                    p = random.uniform(0, 1)
                    if p >= drop_thre:
                        x_collected_words.append(args_data.tag2idx[x_word])
                if len(x_collected_words) >= args_model.max_len:
                    batch_x.append(x_collected_words[:args_model.max_len])
                else:
                    batch_x.append(x_collected_words + [args_model.vocab_size + 1] * (args_model.max_len - len(x_collected_words)))

            batch_y = [c[1] for c in train_set[i:i + batch_range]]

            torch_x = Variable(torch.LongTensor(batch_x)).cuda()
            torch_y = Variable(torch.FloatTensor(batch_y)).cuda()

            model_output = model(torch_x)
            pred = model_output[0]
            log_softmax_output = F.log_softmax(pred, dim=1)
            loss_cls = - torch.sum(log_softmax_output * torch_y) / pred.shape[0]
            loss_l2 = ((model_output[-2] - model_output[-1]) ** 2).mean()
            loss = loss_cls + loss_l2
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=args_hyper.max_norm)
            optimizer.step()

            pred_idx = np.argmax(pred.cpu().data.numpy(), axis=1)
            train_losses.update(loss.item(), batch_range)

            # # FIXME: to top 1 and top-hit
            top1_batch, _, _ = multilabelTop1(pred_idx, batch_y)
            train_top1.update(top1_batch, len(pred_idx))

            hits_batch, _, _ = multilabelHits(pred_idx, batch_y )
            train_hits.update(hits_batch, len(pred_idx))


        dev_top1, dev_hits, dev_loss = test(val_set, model, args_data, args_model)

        if dev_top1 > max_dev_top1:
            max_dev_top1 = dev_top1
            best_model = copy.deepcopy(model)

        if dev_hits > max_dev_hits:
            max_dev_hits = dev_hits



        print('epoch: {} lr: {:.6f}, dev_top1: {:.2f}, dev_hits: {:.2f}, dev_loss: {:.2f}, '
               'train_top1: {:.2f}, train_hits: {:.2f}, train_loss:{:.2f}, max_dev_top1: {:.2f}, '
               'max_dev_hits: {:.2f}'.format(e+1, current_lr,
              dev_top1*100, dev_hits*100, dev_loss, train_top1.avg*100, train_hits.avg*100,
              train_losses.avg,  max_dev_top1*100, max_dev_hits*100))

    print("max dev top1: {:.2f},\tmax dev hits: {:.2f}".format(max_dev_top1, max_dev_hits))
    return best_model


def test(val_set, model, args_data, args_model):
    model.eval()

    # criterion = MultiLabelCrossEntropy()

    x = [[args_data.tag2idx[w] if w in args_data.tag2idx else args_model.vocab_size for w in sent[0][:min(len(sent[0]), args_model.max_len)]] +
         [args_model.vocab_size + 1] * (args_model.max_len - len(sent[0]))
         for sent in val_set]

    y = [c[1] for c in val_set]

    x_torch = Variable(torch.LongTensor(x)).cuda()
    y_torch = Variable(torch.FloatTensor(y)).cuda()

    model_output = model(x_torch)
    pred = model_output[0]

    log_softmax_output = F.log_softmax(pred, dim=1)
    loss = - torch.sum(log_softmax_output * y_torch) / pred.shape[0]

    pred_idx = np.argmax(pred.cpu().data.numpy(), axis=1)

    top1, _, _ = multilabelTop1(pred_idx, y)
    hits, _, _ = multilabelHits(pred_idx, y)

    # largest_values = np.max(y, dim=1)
    # acc_top1 = sum([1 if y[p] == largest_values[i] else 0 for i, (p, y) in enumerate(zip(pred_idx, y))]) / len(pred)
    # acc_exists = sum([1 if y[p] == 1 else 0 for i, (p, y) in enumerate(zip(pred_idx, y))]) / len(pred)
    # cf_matrix = confusion_matrix(y, pred)
    # print(cf_matrix)
    return top1, hits, loss.item()







if __name__ == "__main__":
    main()
