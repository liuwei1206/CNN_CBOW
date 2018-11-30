__author__ = "liuwei"

import torch

from ..function.iniatlize import init_embedding

class WordFeature(torch.nn.Module):
    def __init__(self, vocab_size, word_dim=300, seed=1337, require_grad=True):
        """
        Args:
            vocab_size: the size of word_vocabulary
            word_dim: the size of word_embedding
            require_grad: if true
        """
        super(WordFeature, self).__init__()

        self._vocab_size = vocab_size
        self._word_dim = word_dim
        self._embed = torch.nn.Embedding(vocab_size, word_dim)
        self._embed.weight.requires_grad = require_grad

        # init the weight of embed
        init_embedding(self._embed.weight, seed)

    def forward(self, inputs: torch.LongTensor):
        """
        convert the word indices to the embedding representation
        Args:
            inputs: word indices, [batch_size, n_tokens]
        outputs: [batch_size, n_tokens, word_dim]
        """
        return self._embed(inputs)

class BiasFeature(torch.nn.Module):
    def __init__(self, bias_size, require_grad=True):
        """
        Args:
            bias_size: the size of bias
            require_grad: need grad or not
        """
        super(BiasFeature, self).__init__()

        self._bias_size = bias_size
        self._embed = torch.nn.Embedding(bias_size, 1)
        self._embed.weight.requires_grad = require_grad
        self._embed.weight.data.zero_()

    def forward(self, inputs):
        """
        Args:
            inputs: [sub_size]
        """
        return self._embed(inputs)
