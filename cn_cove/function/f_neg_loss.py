__author__ = "liuwei"

"""
neg loss function
"""
import torch
import numpy as np
from numpy.random import randint
import torch.nn as nn
import math

import datetime
import time
import os


def get_neg_table(vocab_size):
    """
    preduce the neg_table, to get neg_samples
    table_size is 1e8
    every word probability is:
        P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
    """
    neg_table = []
    table_size = 1e8

    vocab_p = []
    for i in range(vocab_size):
        vocab_p.append((math.log(i + 2) - math.log(i + 1)) / math.log(vocab_size + 1))

    now_p = vocab_p[0]
    i = 0
    tt_size = int(table_size)
    for j in range(tt_size):
        neg_table.append(i)
        if (j * 1.0 / table_size) > now_p:
            i += 1
            now_p += vocab_p[i]

    del vocab_p

    return neg_table


def get_neg_samples(neg_size, vocab_size, targets: torch.Tensor,
                    neg_table=None, no_positive=False, unique=False):
    """
    for every batch, get the neg samples, I think don't need so many samples,
    in word2vec, only 5 - 10 every word

    Args:
        neg_size: neg sample how many samples
        targets: positive samples
        no_positive: neg samples didn't include positive samples
        unique: all samples is unique

    """
    table_size = 1e8

    batch_size, n_tokens = targets.size()
    positive_flag = [False] * vocab_size
    unique_flag = [False] * vocab_size

    positive_samples = targets.view(-1).cpu().numpy().astype('long')
    for item in positive_samples:
        positive_flag[item] = True

    if no_positive:
        if unique:
            i = 0
            neg_samples = []
            while i < neg_size:
                sam = randint(low=0,
                              high=1e8,
                              dtype=np.int)

                sam = neg_table[sam]
                if positive_flag[sam]:
                    continue
                elif unique_flag[sam]:
                    continue
                else:
                    neg_samples.append(sam)
                    unique_flag[sam] = True
                    i += 1
        else:
            i = 0
            neg_samples = []
            while i < neg_size:
                sam = randint(low=0,
                              high=1e8,
                              dtype=np.int)
                sam = neg_table[sam]
                if positive_flag[sam]:
                    continue
                else:
                    neg_samples.append(sam)
                    i += 1
    else:
        if unique:
            i = 0
            neg_samples = []
            while i < neg_size:
                sam = randint(low=0,
                              high=1e8,
                              dtype=np.int)
                sam = neg_table[sam]
                if unique_flag[sam]:
                    continue
                else:
                    neg_samples.append(sam)
                    unique_flag[sam] = True
                    i += 1
        else:
            neg_samples = randint(low=0,
                                  high=1e8,
                                  size=(neg_size),
                                  dtype=np.int)
            neg_samples = [neg_table[i] for i in neg_samples]
    neg_samples = np.array(neg_samples, dtype=np.int)

    del positive_flag
    del unique_flag

    neg_samples = np.unique(neg_samples)
    new_targets = np.zeros((batch_size, n_tokens), dtype=np.long)

    return neg_samples, new_targets