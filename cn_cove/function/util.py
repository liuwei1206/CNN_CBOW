__author__ = "liuwei"

"""
common functions
"""

import os
import math
import numpy as np
import datetime
import time
import torch

def time_to_str(timestamp: int):
    """
    convert time to str
    Args:
        timestamp: a linux time, is a long long number

    """
    date_time = datetime.datetime.fromtimestamp(timestamp)
    return '{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}'.format(
        date_time.year, date_time.month, date_time.day,
        date_time.hour, date_time.minute, date_time.second
    )

def str_to_time(time_str: str):
    """
    convert time_str to datetime
    """
    time_str = time_str.split('_')
    d_str = [int(s) for s in time_str[0].split('-')]
    t_str = [int(s) for s in time_str[1].split(':')]

    pieces = d_str + t_str
    return datetime.datetime(*pieces)

def cal_perplexities(loss):
    """
    calculate the perplexities for the loss
    perplexities = e ^ (loss)
    Args:
        loss: the loss value
    """

    return math.exp(loss)

def eval_nearest_neighbors(vocab, embeddings, words, nums=15, gpu=False):
    """
    Args:
        vocab: obj of Vocabulary
        embedding: all word embeddings
        words: a list of chinese words, for eamaple:
               ['中国', '秦始皇', '股票', '狗', '跳舞', '数学', '书法', '一', '奥巴马', '黄河']

    note that, when in train, we generate a batch word ids, all the word-ids has add 1 to do
    mask, so if a word index in vocabulary is 10, then in embedding, its index is 11
    """
    word_ids = [vocab.word_to_id(word) for word in words]
    word_ids = torch.from_numpy(np.array(word_ids)).long()
    if gpu:
        word_ids = word_ids.cuda()

    # size: [words_len, word_dim]
    if torch.cuda.is_available():
        word_ids = word_ids.cuda()
    word_ids_embedding = embeddings(word_ids)
    del word_ids
    
    word_ids_embedding = word_ids_embedding.data.cpu().numpy()
    # weights, [vocab_size, word_dim]
    weights = embeddings._embed.weight
    weights = torch.transpose(weights, 0, 1)
    weights = weights.data.cpu().numpy()

    # [words_len, vocab_size]
    res = np.dot(word_ids_embedding, weights)

    # [words_len, 1]
    word_embedding_L2 = np.linalg.norm(word_ids_embedding, axis=1).reshape(-1, 1)
    # [1, vocab_size]
    weights_L2 = np.linalg.norm(weights, axis=0).reshape(1, -1)
    # [words_len, vocab_size]
    norm_L2 = np.dot(word_embedding_L2, weights_L2)

    # distance = (x * y) / (|x| * |y|)
    res = res / norm_L2

    # sort and get the 15 nearest neighbors
    # remove the padding
    # res = res[:, 1:]

    dt = np.dtype([('index', int), ('value', float)])
    values = []
    x, y = res.shape

    for i in range(x):
        tmp = []
        for j in range(y):
            tmp.append((j, res[i][j]))
        values.append(tmp)

    index_vals = np.array(values, dtype=dt)
    index_vals = np.sort(index_vals, order='value')
    index_vals = np.flip(index_vals, axis=1)

    neighbors = index_vals[:, :nums]
    del index_vals
    del values
    del res

    for i in range(len(words)):
        word = words[i]
        print("{}'s neighbos are:".format(word), end=" ")

        for j in range(nums):
            print(vocab.id_to_word(neighbors[i][j][0]), end=" ")
        print()

    del neighbors












