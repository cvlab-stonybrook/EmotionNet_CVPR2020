import torch
import torch.nn as nn
import torch.nn.functional as F
# a model following DAN


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        # self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1


        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.WV_MATRIX = kwargs["WV_MATRIX"]
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)
        self.fc1 = nn.Linear(self.FILTER_NUM, self.FILTER_NUM)
        self.fc2 = nn.Linear(self.FILTER_NUM, self.CLASS_SIZE)
        # self.fc = nn.Linear(self.FILTER_NUM, self.CLASS_SIZE)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.MAX_SENT_LEN).view(-1, self.FILTER_NUM)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.DROPOUT_PROB, training=self.training)
        x_fc1 = self.fc1(x_avg_drop)
        x_fc1_drop = F.dropout(x_fc1, p=self.DROPOUT_PROB, training=self.training)
        x_final = self.fc2(x_fc1_drop)
        return x_final



class CNN_Embed(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_Embed, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        # self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1


        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        # self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)


    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.MAX_SENT_LEN).view(-1, self.FILTER_NUM)
        x_avg_bn = self.bn(x_avg)

        return x_avg_bn


class CNN_Embed_v2(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_Embed_v2, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = 0

        self.IN_CHANNEL = 1


        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        # self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)
        self.fc1 = nn.Linear(self.FILTER_NUM, self.FILTER_NUM)


    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.MAX_SENT_LEN).view(-1, self.FILTER_NUM)
        x_avg_bn = self.bn(x_avg)
        # x_avg_drop = F.dropout(x_avg_bn, p=self.DROPOUT_PROB, training=self.training)
        x_fc1 = self.fc1(x_avg_bn)
        return x_fc1



class CNN_2Branch(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_2Branch, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = 0.5
        self.IN_CHANNEL = 1


        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if "WV_MATRIX" in kwargs:
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        # self.embedding.weight.requires_grad = False

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)
        self.fc1 = nn.Linear(self.FILTER_NUM, self.FILTER_NUM)
        self.fc2 = nn.Linear(self.FILTER_NUM, self.CLASS_SIZE)
        # self.fc = nn.Linear(self.FILTER_NUM, self.CLASS_SIZE)

    def forward(self, inp):
        raw_feature = self.embedding(inp)
        x = raw_feature.view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.MAX_SENT_LEN).view(-1, self.FILTER_NUM)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.DROPOUT_PROB, training=self.training)
        x_fc1 = self.fc1(x_avg_drop)
        x_fc1_drop = F.dropout(x_fc1, p=self.DROPOUT_PROB, training=self.training)
        x_final = self.fc2(x_fc1_drop)
        return x_final, x_fc1


class CNN_Word_Embed_BN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_Word_Embed_BN, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        # self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        # self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.MAX_SENT_LEN).view(-1, self.FILTER_NUM)
        x_avg_bn = self.bn(x_avg)

        return x_avg_bn

class CNN_Word_Embed(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_Word_Embed, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        # self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        # self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feat = F.relu(self.conv(x))
        return x_feat



class CNN_SelfAttention(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_SelfAttention, self).__init__()

        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = 0
        self.IN_CHANNEL = 1



        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.WV_MATRIX = kwargs["WV_MATRIX"]
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM, self.WORD_DIM, stride=self.WORD_DIM)
        self.bn = nn.BatchNorm1d(self.FILTER_NUM)
        self.fc1 = nn.Linear(self.FILTER_NUM, self.FILTER_NUM)
        self.fc2 = nn.Linear(self.FILTER_NUM, self.CLASS_SIZE)
        # self.fc = nn.Linear(self.FILTER_NUM, self.CLASS_SIZE)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        x_feature = self.embedding(inp)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.MAX_SENT_LEN).view(-1, self.FILTER_NUM)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.DROPOUT_PROB, training=self.training)
        x_fc1 = self.fc1(x_avg_drop)
        x_fc1_drop = F.dropout(x_fc1, p=self.DROPOUT_PROB, training=self.training)
        x_final = self.fc2(x_fc1_drop)
        return x_final, x_feature