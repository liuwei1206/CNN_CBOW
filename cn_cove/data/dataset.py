__author__ = "liuwei"

"""
read a batch of data each iter, the data is word_ids or char_ids,
and include the reverse data, the targets of data
"""

import os
import torch
import numpy as np
from typing import List, Dict
import random
import copy

def batch_to_word_ids(batch_word, vocab, reverse=False, split=False):
    """
    Args:
        vocab: word vocabulary
        batch: one batch str data, include batch_size sentence
        reverse:
        split:
    inputs: [batch_size, None]
    outputs: [batch_size, n_tokens]
    """
    batch_size = len(batch_word)
    max_sent_len = max([len(sent) for sent in batch_word])
    max_sent_len += 2
    X_word_ids = np.zeros((batch_size, max_sent_len), dtype=np.int32)

    for k, sent in enumerate(batch_word):
        length = len(sent) + 2
        ids_without_mask = vocab.encode(sent, reverse, split)
        # add 1 for mask
        X_word_ids[k, :length] = ids_without_mask + 1

    return X_word_ids

class LM_Dataset():
    def __init__(self, datadir, batch_size=32, vocab=None):
        """
        Args:
            datadir: data file dir
            vocab: Vocabulary object
        """
        super(LM_Dataset, self).__init__()

        files = os.listdir(datadir)
        self._files = [datadir + '/' + file for file in files]
        self._batch_size = 32
        self.vocab = vocab

    def get_batch(self):
        """
        read a batch data
        """
        random.shuffle(self._files)
        batch_size = self._batch_size
        for file in self._files:
            with open(file, 'r') as f:
                lines = f.readlines()
                # shuffle the datas
                random.shuffle(lines)
                data_len = len(lines)
                batch_num = int(data_len // batch_size)

                for j in range(batch_num):
                    batch_sents = lines[j * batch_size: (j+1) * batch_size]
                    batch_data = []
                    for sent in batch_sents:
                        batch_data.append([word for word in sent.split()])
                    del batch_sents
                    '''
                    reverse_batch_word = copy.deepcopy(batch_data)
                    _ = [sent.reverse() for sent in reverse_batch_word]

                    batch_word_ids = batch_to_word_ids(batch_data, self.vocab)
                    reverse_batch_word_ids = batch_to_word_ids(reverse_batch_word, self.vocab)

                    batch_word_ids = torch.from_numpy(batch_word_ids).long()
                    reverse_batch_word_ids = torch.from_numpy(reverse_batch_word_ids).long()
                    '''

                    batch_word_ids = batch_to_word_ids(batch_data, self.vocab)
                    del batch_data
                    reverse_batch_word_ids = copy.deepcopy(batch_word_ids)
                    reverse_batch_word_ids = np.flip(reverse_batch_word_ids, axis=1).copy()

                    batch_word_ids = torch.from_numpy(batch_word_ids).long()
                    reverse_batch_word_ids = torch.from_numpy(reverse_batch_word_ids).long()

                    batch_inputs = {}
                    batch_inputs['batch_word_ids'] = batch_word_ids
                    batch_inputs['reverse_batch_word_ids'] = reverse_batch_word_ids

                    del batch_word_ids
                    del reverse_batch_word_ids

                    yield batch_inputs
