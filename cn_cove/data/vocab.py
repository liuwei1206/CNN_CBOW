__author__ = "liuwei"

import os
import numpy as np
from typing import List, Dict

def make_vocab_file(data_dir, min_cnt=50):
    """
    Args:
        data_dir: the dir of all train data files
        min_cnt: min word frequence to be remove out from the vocab
    """
    files = os.listdir(data_dir)
    files = [data_dir + '/' + file for file in files]
    vocab_data = dict()

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                for w in line.split():
                    if w in vocab_data:
                        vocab_data[w] += 1
                    else:
                        vocab_data[w] = 1

    #sorted the word use word frequence
    vocab_data = sorted(vocab_data.items(), key=lambda item: item[1], reverse=True)
    vocab_data = [(item[0]) for item in vocab_data if item[1] >= min_cnt]
    print('size of the vocabulary is: ', len(vocab_data))

    with open('data/vocab.txt', 'w') as f:
        f.write("<S>\n")
        f.write("</S>\n")
        f.write("<UNK>\n")

        vocab_data = [s + '\n' for s in vocab_data]
        f.writelines(vocab_data)

    del vocab_data


def get_vocab_size(vocab_file):
    """
    Args:
        vocab_file: the path of vocab.txt
    """
    with open(vocab_file, 'r') as f:
        return len(f.readlines())


def get_data_rows(data_dir):
    """
    get the rows of all train data file
    """
    files = os.listdir(data_dir)
    files = [data_dir + '/' + file for file in files]

    all_rows = 0
    for file in files:
        with open(file, 'r') as f:
            all_rows += len(f.readlines())

    return all_rows

def words_from_vocab(vocab_file):
    """
    read words from the vocab_file, the word in vocab_file is one word per line.
    Note that special words such as '<S>', '</S>', '<UNK>' need add to vocab
    Args:
        vocab_file: the file path of vocab file
    """
    words = []
    with open(vocab_file, 'r') as f:
        for line in f:
            words.append(line.strip())

    return words

def get_batch_num_by_words(data_dir, batch_words_num):
    """
    get the total batch number of train corpus
    Args:
        data_dir: the dir include train files
    """
    files = os.listdir(data_dir)
    files = [data_dir + '/' + file for file in files]

    total_batch = 0
    for file in files:
        words_this_file = 0
        with open(file, 'r') as f:
            for line in f:
                words_this_file += len(line.strip().split())
        batch_this_file = (words_this_file - 2) // batch_words_num

        total_batch += batch_this_file

    return total_batch

class Vocabulary(object):
    """
    class of word vocabulary
    """
    def __init__(self, vocab_file):
        super(Vocabulary, self).__init__()

        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        # one word one line
        with open(vocab_file, 'r') as f:
            idx = 0
            for line in f:
                word_text = line.strip()
                if word_text == '<S>':
                    self._bos = idx
                elif word_text == '</S>':
                    self._eos = idx
                elif word_text == '<UNK>':
                    self._unk = idx
                elif word_text == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_text)
                self._word_to_id[word_text] = idx
                idx += 1

    @property
    def size(self):
        return len(self._id_to_word)
    @property
    def bos(self):
        return self._bos
    @property
    def eos(self):
        return self._eos

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self._unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        """
        Args:
            sentence: string or list[str]
            reverse: bool, is the sentence has been reserved or not
            split: bool, sentence is a list of data or a string, if true, mean need split
        """
        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence
            ]

        if reverse:
            return np.array([self._eos] + word_ids + [self._bos])
        else:
            return np.array([self._bos] + word_ids + [self._eos])

