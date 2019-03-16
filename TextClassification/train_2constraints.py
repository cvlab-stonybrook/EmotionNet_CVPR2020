# Copyright (c) 2019 Zijun Wei.
# Licensed under the MIT License.
# Author: Zijun Wei
# Usage(TODO): modified from run_completeAMT_mclass_fixedlength.py in ~/Dev/TextClassification
# Email: hzwzijun@gmail.com
# Created: 02/Mar/2019 18:43

from TextClassification.model_DAN_2constraints import CNN

import TextClassification.utils as utils
from collections import Counter

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
import TextClassification.load_data as read_AMT_data
import torch.nn.functional as F
import random
from torch.optim import lr_scheduler
import tqdm
from CNNs.utils.util import AverageMeter




def test_690_contain(tag2idx):
    emotion690_vocabulary = loadpickle('/home/zwei/Dev/AttributeNet3/AdobeStockSelection/RetrieveSelected778/data_v6_690_xmas/Emotion_vocabulary.pkl')
    selected_emotion690 = emotion690_vocabulary['key2idx']

    all_found = True
    for x in selected_emotion690:
        if x not in tag2idx:
                print("{} Not Found".format(x))
                all_found = False
    if all_found:
        print("All the 690 words can be found in this dict")



def main():
    parser = argparse.ArgumentParser(description="CNN Text Classification with Pytorch")
    parser.add_argument("--epoch", default=80, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", '-lr', default=1.0, type=float, help="learning rate")
    parser.add_argument("--lr_schedule", '-lrs', default='10, 20, 30', type=str, help="the schedule of learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--word_dim', default=300, type=int, help="word dimension")
    parser.add_argument('--filter_dim', default=300, type=int, help="intermediate feature size, same as the word-dim will make it much easier")
    parser.add_argument('--sent_len', default=100, type=int, help="length of each sentence, will set to max length if not given")
    parser.add_argument('--batch_size', default=50, type=int, help="batch-size")
    parser.add_argument('--save_to', default='models/model_feature_regularization.pth.tar')
    args = parser.parse_args()



    data = {}
    train_set, val_set = read_AMT_data.read_AMT_complete_mtrain_stest()

    data["vocab"] = sorted(list(set([w for sent in train_set + val_set for w in sent[0]])))
    data["classes"] = [0, 1, 2, 3, 4, 5, 6, 7]
    data["tag2idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx2tag"] = {i: w for i, w in enumerate(data["vocab"])}

    test_690_contain(data['tag2idx'])

    params = {
        "EPOCH": args.epoch,
        "Learning_SCHEDULE": [int(i) for i in args.lr_schedule.split(',')],
        "LEARNING_RATE": args.learning_rate,
        "MAX_SENT_LEN": args.sent_len or max([len(sent[0]) for sent in train_set + val_set]),
        "BATCH_SIZE": args.batch_size,
        "WORD_DIM": args.word_dim,
        "FILTER_NUM": args.filter_dim,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": args.gpu,
        "tag2idx": data['tag2idx'],
        "idx2tag": data['idx2tag'],
        'save_path': args.save_to
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("MAX_SENT_SIZE:", params["MAX_SENT_LEN"])
    print("WORD_DIM:", params["WORD_DIM"])
    print("FILTER_NUM:", params["FILTER_NUM"])

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    best_model = train(train_set, val_set, data, params)
    # evaluate_best_model(val_set, data, best_model, params)

    utils.save_model(best_model, params)
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)




def train(train_set, val_set, data_info, params, embedding_size=300):
    # print("loading word2vec...")
    pretrained_w2v = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/pre_extract_w2v/params/selftrained_extracted_w2v_wordnet_synsets_py3.pl')
    # pretrained_w2v = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/pre_extract_w2v/params/googlenews_extracted_w2v_wordnet_synsets_py3.pl')

    #
    wv_matrix = []
    words_not_found = []
    for i in range(len(data_info["vocab"])):
        word = data_info["idx2tag"][i]
        if word in pretrained_w2v:
            wv_matrix.append(pretrained_w2v[word])
        else:
            words_not_found.append(word)
            # print("{} not found in dictrionary, will use random".format(word))
            wv_matrix.append(np.random.uniform(-0.01, 0.01, embedding_size).astype("float32"))
    print("{} words were not found".format(len(words_not_found)))
    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, embedding_size).astype("float32"))
    wv_matrix.append(np.zeros(embedding_size).astype("float32"))
    wv_matrix = np.array(wv_matrix)
    params["WV_MATRIX"] = wv_matrix

    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, params['Learning_SCHEDULE'], gamma=0.1)
    max_dev_acc = 0
    max_test_acc = 0
    best_model = None
    model.train()

    for e in range(params["EPOCH"]):
        train_set = shuffle(train_set)
        train_losses = AverageMeter()
        train_accuracies = AverageMeter()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        model.train()

        for i in tqdm.tqdm(range(0, len(train_set), params["BATCH_SIZE"])):
            batch_range = min(params["BATCH_SIZE"], len(train_set) - i)

            # add random dropping words:
            batch_x = []

            for sent in train_set[i:i + batch_range]:
                x_sent = sent[0]
                drop_thre = 0.2
                x_collected_words = []
                for x_word in x_sent:
                    p = random.uniform(0, 1)
                    if p >= drop_thre:
                        x_collected_words.append(data_info["tag2idx"][x_word])
                if len(x_collected_words) >= params["MAX_SENT_LEN"]:
                    batch_x.append(x_collected_words[:params["MAX_SENT_LEN"]])
                else:
                    batch_x.append(x_collected_words + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(x_collected_words)))

            batch_y_numpy = [c[1] for c in train_set[i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.FloatTensor(batch_y_numpy)).cuda(params["GPU"])

            optimizer.zero_grad()
            pred, raw_feature, transfromed_feature = model(batch_x)
            # loss = criterion(pred, batch_y)

            log_softmax_output = F.log_softmax(pred, dim=1)

            loss_cls = - torch.sum(log_softmax_output * batch_y) / pred.shape[0]
            loss_l2 = ((raw_feature - transfromed_feature)**2).mean()
            loss = loss_cls + loss_l2

            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()
            pred = np.argmax(pred.cpu().data.numpy(), axis=1)

            acc = sum([1 if y[p] > 0 else 0 for p, y in zip(pred, batch_y_numpy)]) / len(pred)
            train_losses.update(loss.item(), batch_range)
            train_accuracies.update(acc, batch_range)


        dev_acc, dev_loss = test(val_set ,data_info, model, params, criterion)

        # if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
        #     print("early stopping by dev_acc!")
        #     break
        # else:
        #     pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            best_model = copy.deepcopy(model)

        print("epoch:", e + 1, ' lr:', '{:.6f}'.format(current_lr),  " dev_acc:", dev_acc, ' dev_loss:', dev_loss, " train_acc:", train_accuracies.avg, ' train_losses:', train_losses.avg, ' max_dev_acc ', max_dev_acc)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def test(data_x, data, model, params, criterion):
    model.eval()

    x = [[data["tag2idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent[0][:min(len(sent[0]), params["MAX_SENT_LEN"])]] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent[0]))
         for sent in data_x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [c[1] for c in data_x]
    batch_y = Variable(torch.LongTensor(y)).cuda(params["GPU"])

    pred, _, _ = model(x)
    loss = criterion(pred, batch_y)

    pred = np.argmax(pred.cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    cf_matrix = confusion_matrix(y, pred)
    print(cf_matrix)
    return acc, loss.item()

def evaluate_best_model(data_x, data, model, params):
        model.eval()
        image_data = loadpickle('/home/zwei/Dev/TextClassifications/z_implementations/AMT_data/processed/mturk_annotations.pkl')

        x = []

        for sent in data_x:
            x_sent = sent[0]
            x_collected_words = []
            for x_word in x_sent:
                    x_collected_words.append(data["tag2idx"][x_word])
            if len(x_collected_words) >= params["MAX_SENT_LEN"]:
                x.append(x_collected_words[:params["MAX_SENT_LEN"]])
            else:
                x.append(
                    x_collected_words + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(x_collected_words)))

        x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
        y = [c[1] for c in data_x]
        batch_y = Variable(torch.LongTensor(y)).cuda(params["GPU"])

        pred, _, _ = model(x)
        pred = F.softmax(pred, dim=1).cpu().data.numpy()
        pred_idx = np.argmax(pred, axis=1)
        # cf_matrix = confusion_matrix(y, pred_idx)
        # print(cf_matrix)
        correct_happiness_count = 0
        correct_neutral_count = 0
        for idx in range(len(data_x)):
            # if idx % 100 == 0:
            if pred_idx[idx] == y[idx] and (pred_idx[idx] == 3 or pred_idx[idx] == 4):
                # print("Correctly Classified")
                if pred_idx[idx] == 3:
                    correct_happiness_count += 1

                else:
                    correct_neutral_count += 1


            else:

                print("image_cid: {}".format(data_x[idx][2]))
                print('{}'.format(image_data[data_x[idx][2]]['url']))
                print('labels:', ', '.join(x for x in data_x[idx][0]))
                print(read_AMT_data.idx2emotion)
                print("pred: ", ', '.join(str(x) for x in pred[idx]))
                print("pred class: {}, {}".format(read_AMT_data.idx2emotion[pred_idx[idx]], pred_idx[idx]))
                print("gt class: {}, {}".format(read_AMT_data.idx2emotion[y[idx]], y[idx]))
                if pred_idx[idx] == y[idx]:
                    print("Correctly Classified")
                else:
                    print("Wrongly Classified")

                s_annotated_emotions = image_data[data_x[idx][2]]['image_emotion']
                s_emotions = []
                for s_s_emotion in s_annotated_emotions:
                    s_emotions.extend(s_s_emotion)
                s_emotions = Counter(s_emotions)

                print("raw annotations {}".format(', '.join(['{}({})'.format(x[0], x[1]) for x in s_emotions.most_common()])) )

        print("Correct Happiness/Neutral count: {}/{}".format(correct_happiness_count, correct_neutral_count))







if __name__ == "__main__":
    main()
