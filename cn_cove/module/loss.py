__author__ = "liuwei"

"""
use it to calculate the loss, provide two kind of loss
fullsoftmax and neg_loss
"""
import torch
import numpy as np
import torch.nn as nn

from ..function.f_neg_loss import get_neg_samples


class Full_Loss(torch.nn.Module):
    def __init__(self, loss, softmax_weight, softmax_bias, gpu=True):
        """
        Args:
            loss: loss function
            softmax_weight: the softmax layer weights
            softmax_bias: the softmax layer bias
        """
        super(Full_Loss, self).__init__()
        
        self._gpu = gpu
        self._loss = loss
        self._weight = torch.transpose(softmax_weight._embed.weight, 0, 1)
        self._bias = torch.transpose(softmax_bias._embed.weight, 0, 1)
        if gpu:
            self._weight = self._weight.cuda()
            self._bias = self._bias.cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: the context of each word, [batch_size, n_tokens, word_dim]
            targets: the target word, [batch_size, n_tokens]
        """
        # cal the probability of every predict class
        # [batch_size, n_tokens, vocab_size]
        outputs = torch.matmul(inputs, self._weight) + self._bias
        _, predicts = torch.max(outputs, -1)

        # cal crossentropy
        outputs = torch.transpose(outputs, 1, 2)
        loss = self._loss(outputs, targets)
        del outputs

        # cal accu
        res = (predicts.float() - targets.float())
        del predicts
        total_size = targets.size()[0] * targets.size()[1]
        accu = (res == 0).float().sum() / total_size
        del res

        return loss, accu

class Neg_Loss(torch.nn.Module):
    def __init__(self, loss, neg_size, vocab_size, neg_table, softmax_weight,
                 softmax_bias, no_positive=True, unique=True, gpu=True):
        """
        Args:
            loss: loss function
            neg_size: the size of neg samples
            vocab_size: the size of vocabulary
            neg_table: neg probabily table
            softmax_weight:
            softmax_bias:
            is_positive: neg samples don't include positive samples
            unique: neg samples are unique
            gpu: has gpu or not
        """
        super(Neg_Loss, self).__init__()

        self._loss = loss
        self._neg_size = neg_size
        self._vocab_size = vocab_size
        self._neg_table = neg_table
        self._softmax_weight = softmax_weight
        self._softmax_bias = softmax_bias
        self._no_positive = no_positive
        self._unique = unique
        self._gpu = gpu

    def forward(self, inputs, targets):
        """
        Args:
            inputs: the context of each token, size is [batch_size, n_tokens, word_dim]
            targets: target tokens size is [batch_size, n_tokens]

        return loss:
            loss:
            accu:
        """
        # get the subset, subset = (target_token + neg_samples)
        neg_samples, labels = get_neg_samples(self._neg_size, self._vocab_size, targets,
                                      self._neg_table, self._no_positive, self._unique)

        neg_samples = torch.from_numpy(neg_samples)
        labels = torch.from_numpy(labels)
        if self._gpu:
            neg_samples = neg_samples.cuda()

        # cal true logits
        # inputs: [batch_size, n_tokens, word_dim]
        # targets_softmax_weight: [batch_size, n_token, word_dim],
        # targets_softmax_bias: [batch_size, n_tokens, 1]
        targets_softmax_weight = self._softmax_weight(targets)
        targets_softmax_bias = self._softmax_bias(targets)
        # true logits, shape: [batch_size, n_tokens, 1]
        true_logits = inputs * targets_softmax_weight
        true_logits = torch.sum(true_logits, dim=-1, keepdim=True) + targets_softmax_bias
        del targets_softmax_weight
        del targets_softmax_bias

        # cal neg_samples logits,
        # inputs: [batch_size, n_tokens, word_dim]
        # neg_softmax_weights:[word_dim, neg_samples_size],
        # neg_softmax_bias: [1, neg_samples_size]
        neg_softmax_weight = self._softmax_weight(neg_samples)
        neg_softmax_weight = torch.transpose(neg_softmax_weight, 0, 1)
        neg_softmax_bias = self._softmax_bias(neg_samples)
        neg_softmax_bias = torch.transpose(neg_softmax_bias, 0, 1)
        del neg_samples
        # neg logits: [batch_size, n_tokens, neg_samples_size]
        neg_logits = torch.matmul(inputs, neg_softmax_weight) + neg_softmax_bias
        del neg_softmax_weight
        del neg_softmax_bias

        # logits: [batch_size, n_tokens, neg_samples_size + 1]
        logits = torch.cat((true_logits, neg_logits), dim=-1)
        del true_logits
        del neg_logits
        _, predicts = torch.max(logits, -1)

        logits = torch.transpose(logits, 1, 2)
        # labels
        if self._gpu:
            labels = labels.cuda()

        # loss, [batch_size, neg_samples_size+1, n_tokens], [batch_size, n_tokens]
        loss = self._loss(logits, labels)
        del logits

        # cal accu
        total_size = labels.size()[0] * labels.size()[1]
        res = (predicts == 0).float()
        accu = res.sum() / total_size
        #accu = (predicts == 0).float().sum() / total_size
        del res
        del predicts
        del labels

        return loss, accu

