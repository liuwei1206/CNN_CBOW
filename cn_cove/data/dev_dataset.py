__author__ = "liuwei"

"""
the dataset generator of validation 
"""

import numpy as np
import torch
import os
import random
import copy

from .dataset import batch_to_word_ids

class Validation_Dataset():
    def __init__(self, data_dir, batch_size=32, vocab=None):
        """
        Args:
            data_dir: the data dir contain the validation datas
            vocab: Vocabulary obj
        """
        super(Validation_Dataset, self).__init__()

        files = os.listdir(data_dir)
        self._files = [data_dir + '/' + file for file in files]
        self._batch_size = batch_size
        self.vocab = vocab

    def get_batch(self):
        """
        only use 10 batch to validation the model, every epoch we random select a file,
        and shuffle all the data, and get 10 batch data
        """
        batch_size = self._batch_size
        file_num = len(self._files) - 1
        selected_file_num = random.randint(0, file_num)
        selected_file = self._files[selected_file_num]
        count = 10

        with open(selected_file, 'r') as f:
            lines = f.readlines()

            random.shuffle(lines)
            #data_len = len(lines)
            #batch_num = int(data_len // batch_size)
            for j in range(count):
                batch_sents = lines[j * batch_size: (j + 1) * batch_size]
                batch_data = []
                for sent in batch_sents:
                    batch_data.append([word for word in sent.split()])
                del batch_sents
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



