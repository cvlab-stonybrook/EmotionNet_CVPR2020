import torch
import torch.nn as nn
import torch.nn.functional as F
# a model following DAN


class TextCNN(nn.Module):
    def __init__(self, args, init_wv=None):
        super(TextCNN, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size + 2, self.args.word_dim, padding_idx=self.args.vocab_size + 1)
        if init_wv is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_wv))
        if self.args.static:
            self.embedding.weight.requires_grad = False

        self.conv = nn.Conv1d(self.args.in_channel, self.args.filter_num, self.args.word_dim, stride=self.args.word_dim)
        self.bn = nn.BatchNorm1d(self.args.filter_num)
        self.fc1 = nn.Linear(self.args.filter_num, self.args.filter_num)
        self.fc2 = nn.Linear(self.args.filter_num, self.args.num_classes)

    def forward(self, word_idx):
        raw_feature = self.embedding(word_idx)
        x = raw_feature.view(-1, 1, self.args.word_dim * self.args.max_len)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.args.max_len).view(-1, self.args.filter_num)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.args.dropout_prob, training=self.training)
        x_fc1 = self.fc1(x_avg_drop)
        x_fc1_drop = F.dropout(x_fc1, p=self.args.dropout_prob, training=self.training)
        x_final = self.fc2(x_fc1_drop)
        return x_final, x_fc1, x_avg_bn, x_avg, x_feat.permute(0, 2, 1), raw_feature


class TextCNN_LT(TextCNN):
    def __init__(self, args, init_wv=None):
        super(TextCNN_LT, self).__init__(args, init_wv)


    def forward(self, word_idx):
        raw_feature = self.embedding(word_idx)
        # x = raw_feature.view(-1, 1, self.args.word_dim * self.args.max_len)
        x_feat = raw_feature.permute(0, 2, 1)
        x_avg = F.avg_pool1d(x_feat, self.args.max_len).view(-1, self.args.filter_num)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.args.dropout_prob, training=self.training)
        x_fc1 = self.fc1(x_avg_drop)
        x_fc1_drop = F.dropout(x_fc1, p=self.args.dropout_prob, training=self.training)
        x_final = self.fc2(x_fc1_drop)
        return x_final, x_fc1, x_avg_bn, x_avg, x_feat.permute(0, 2, 1), raw_feature


class TextCNN_Avg(nn.Module):
    def __init__(self, args, init_wv=None):
        super(TextCNN_Avg, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size + 2, self.args.word_dim, padding_idx=self.args.vocab_size + 1)
        if init_wv is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_wv))
        if self.args.static:
            self.embedding.weight.requires_grad = False

        self.bn = nn.BatchNorm1d(self.args.word_dim)
        self.fc = nn.Linear(self.args.word_dim, self.args.num_classes)

    def forward(self, word_idx):
        raw_feature = self.embedding(word_idx)
        x_avg = F.avg_pool1d(raw_feature.permute(0, 2, 1), self.args.max_len).view(-1, self.args.word_dim)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.args.dropout_prob, training=self.training)
        x_final = self.fc(x_avg_drop)
        return x_final, x_avg_bn, x_avg, raw_feature


class TextCNN_NLT(nn.Module):
    def __init__(self, args, init_wv=None):
        super(TextCNN_NLT, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size + 2, self.args.word_dim, padding_idx=self.args.vocab_size + 1)
        if init_wv is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_wv))
        if self.args.static:
            self.embedding.weight.requires_grad = False
        self.conv = nn.Conv1d(self.args.in_channel, self.args.filter_num, self.args.word_dim, stride=self.args.word_dim)

        self.bn = nn.BatchNorm1d(self.args.word_dim)
        self.fc = nn.Linear(self.args.word_dim, self.args.num_classes)

    def forward(self, word_idx):
        raw_feature = self.embedding(word_idx)

        x = raw_feature.view(-1, 1, self.args.word_dim * self.args.max_len)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.args.max_len).view(-1, self.args.filter_num)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.args.dropout_prob, training=self.training)
        x_final = self.fc(x_avg_drop)
        return x_final, x_avg_bn, x_avg, x_feat.permute(0, 2, 1), raw_feature



class TextCNN_NLT_DAN(nn.Module):
    def __init__(self, args, init_wv=None):
        super(TextCNN_NLT_DAN, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size + 2, self.args.word_dim, padding_idx=self.args.vocab_size + 1)
        if init_wv is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(init_wv))
        if self.args.static:
            self.embedding.weight.requires_grad = False
        self.conv = nn.Conv1d(self.args.in_channel, self.args.filter_num, self.args.word_dim, stride=self.args.word_dim)

        self.bn = nn.BatchNorm1d(self.args.word_dim)
        self.fc_m = nn.Linear(self.args.word_dim, self.args.word_dim)
        self.fc = nn.Linear(self.args.word_dim, self.args.num_classes)

    def forward(self, word_idx):
        raw_feature = self.embedding(word_idx)

        x = raw_feature.view(-1, 1, self.args.word_dim * self.args.max_len)
        x_feat = F.relu(self.conv(x))
        x_avg = F.avg_pool1d(x_feat, self.args.max_len).view(-1, self.args.filter_num)
        x_avg_bn = self.bn(x_avg)
        x_avg_drop = F.dropout(x_avg_bn, p=self.args.dropout_prob, training=self.training)
        x_m = self.fc_m(x_avg_drop)
        x_m_drop = F.dropout(x_m, p=self.args.dropout_prob, training=self.training)
        x_final = self.fc(x_m_drop)
        return x_final, x_m, x_avg_bn, x_avg, x_feat.permute(0, 2, 1), raw_feature


# class TextCNN_Z(nn.Module):
#     def __init__(self, args, init_wv=None):
#         super(TextCNN_Z, self).__init__()
#
#         self.args = args
#
#         self.embedding = nn.Embedding(self.args.vocab_size + 2, self.args.word_dim,
#                                       padding_idx=self.args.vocab_size + 1)
#
#         # self.static_embed = nn.Embedding(self.args.vocab_size + 2, self.args.word_dim,
#         #                               padding_idx=self.args.vocab_size + 1)
#         if init_wv is not None:
#             self.embedding.weight.data.copy_(torch.from_numpy(init_wv))
#             # self.static_embed.weight.data.copy_(torch.from_numpy(init_wv))
#             # self.static_embed.weight.requires_grad = False
#         if self.args.static:
#             self.embedding.weight.requires_grad = False
#
#         self.trans = nn.Linear(self.args.word_dim, self.args.word_dim)
#         self.bn = nn.BatchNorm1d(self.args.word_dim)
#         # self.fc_m = nn.Linear(self.args.word_dim, self.args.word_dim)
#         self.fc = nn.Linear(self.args.word_dim, self.args.num_classes)
#
#     def forward(self, word_idx):
#         # static_feature = self.static_embed(word_idx)
#         raw_feature = self.embedding(word_idx)
#         trans_feature = self.trans(raw_feature.view(-1, self.args.word_dim)).view_as(raw_feature)
#         x_avg = torch.mean(raw_feature, dim=1)
#         x_avg_bn = self.bn(x_avg)
#         x_avg_drop = F.dropout(x_avg_bn, p=self.args.dropout_prob, training=self.training)
#         # x_m = self.fc_m(x_avg_drop)
#         # x_final = F.dropout(self.fc(x_m), p=self.args.dropout_prob, training=self.training)
#         x_final = self.fc(x_avg_drop)
#         return x_final, x_avg_bn, x_avg, trans_feature, raw_feature
