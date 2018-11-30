__author__ = "liuwei"

"""
a new dataset for train. this is Inspired by the data_reader in 
'https://github.com/mkroutikov/tf-lstm-char-cnn'.

Read out the all data in the file, and put all the words in a single 
array, when we generate a batch, we read a number of batch_size * n_tokens
data, then there is nearly no padding in the batch, which is very good.

"""
import os
import torch
import numpy as np
from typing import List, Dict
import random
import copy

class Train_Dataset():
    def __init__(self, datadir, batch_size=32, n_tokens=120, windom_size=10,
                 vocab=None, gpu=False, count=None):
        """
        Args:
            datadir: data file dir
            vocab: Vocabulary object
            batch_size: the size of batch
            n_tokens: every row's token number
            windom_size: the windom_size of cnn
            gpu: has gpu or not
        """
        super(Train_Dataset, self).__init__()

        files = os.listdir(datadir)
        self._files = [datadir + '/' + file for file in files]
        self._batch_size = batch_size
        self._n_tokens = n_tokens
        self._windom_size = windom_size
        self.vocab = vocab
        self._gpu = gpu
        self._count = count

    def get_file_words(self, file):
        """
        read all the data in the file into a array.
        Args:
            file: the data file path
        """
        file_words = []

        with open(file, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                line = line.strip()
                words = line.split()

                # note add the begin of sentence and end of sentence word
                file_words.append(self.vocab.bos)
                for word in words:
                    file_words.append(self.vocab.word_to_id(word))
                file_words.append(self.vocab.eos)

        del lines

        return file_words


    def get_batch(self):
        """
        read a batch data for train
        """
        batch_size = self._batch_size
        random.shuffle(self._files)

        for file in self._files:
            # get all words
            file_words = self.get_file_words(file)
            # total words number, and total batch number, each batch include batch_size * n_tokens
            word_len = len(file_words)
            each_batch_num = self._batch_size * self._n_tokens

            # the data that match batch number
            reduce_file_length = (word_len - 2 // each_batch_num) * each_batch_num
            total_batch = (word_len - 2) // each_batch_num
            file_words = file_words[:reduce_file_length + 2]

            # count control the dataset use to train or dev
            if self._count is not None and self._count < total_batch:
                total_batch = self._count

            for j in range(total_batch):
                # target_ids, forward_word_ids, backword_word_ids
                batch_target_ids = file_words[1 + j * each_batch_num: 1 + (j + 1) * each_batch_num]
                batch_word_ids = file_words[j * each_batch_num: (j + 1) * each_batch_num]

                reverse_batch_word_ids = file_words[2 + j * each_batch_num: 2 + (j + 1) * each_batch_num]
                reverse_batch_word_ids = copy.deepcopy(reverse_batch_word_ids)
                reverse_batch_word_ids.reverse()

                # reshape to [batch_size, n_tokens]
                batch_word_ids, batch_target_ids = np.array(batch_word_ids), np.array(batch_target_ids)
                reverse_batch_word_ids = np.array(reverse_batch_word_ids)
                batch_word_ids = batch_word_ids.reshape((-1, self._n_tokens))
                batch_target_ids = batch_target_ids.reshape((-1, self._n_tokens))

                reverse_batch_word_ids = reverse_batch_word_ids.reshape((-1, self._n_tokens))
                reverse_batch_word_ids = np.flipud(reverse_batch_word_ids).copy()

                assert batch_word_ids.shape[0] == self._batch_size

                batch_word_ids = torch.from_numpy(batch_word_ids).long()
                reverse_batch_word_ids = torch.from_numpy(reverse_batch_word_ids).long()
                batch_target_ids = torch.from_numpy(batch_target_ids)

                batch_inputs = {}
                batch_inputs['batch_word_ids'] = batch_word_ids
                batch_inputs['reverse_batch_word_ids'] = reverse_batch_word_ids
                batch_inputs['batch_target_ids'] = batch_target_ids

                del batch_word_ids
                del reverse_batch_word_ids
                del batch_target_ids

                yield batch_inputs


    def get_good_batch(self):
        """

        """