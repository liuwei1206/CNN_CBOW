__author__ = "liuwei"

"""
a language model, seem to word2vec, use cnn to capture context

we fix the window size, default is 10. get both the left and right
context of the word. and then concat the left and right to predict 
the center word.
to guarantee different convolution get diffrent msg, we must init 
the convolution diffrently
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..function.iniatlize import init_cnn_weight, init_linear
from ..function.get_padding import get_padding
from ..module.feature import WordFeature
from ..module.highway import Highway

class CNN_LM(torch.nn.Module):
    def __init__(self, word_dim=300, n_filters=300, n_cnns=2,
                 window=10, drop_pro=0.5, n_highway=1, gpu=False,
                 require_grad=True):
        """
        Args:
            word_dim: the dim of word embedding
            n_filters: the output dim of cnn
            n_cnns: the num of cnn layers
            window: the size of context window
            drop_pro: the probalility of dropout
            n_highway: the number of highway layers
            require_grad:
        """
        super(CNN_LM, self).__init__()

        self._word_dim = word_dim
        self._n_filters = n_filters
        self._n_cnns = n_cnns
        self._window = window
        self._n_highway = n_highway
        self._gpu = gpu
        self._require_grad = require_grad

        self._dropout = nn.Dropout(p=drop_pro)
        # init net and padding
        # self._paddings = {}
        self.build_cnn()
        self.build_proj()

        # hightway
        if n_highway > 0:
            self._highway = Highway(word_dim * 2, n_highway, require_grad=require_grad)

    def build_cnn(self):
        """
        we use conv1d to capture context
        """
        for i in range(self._n_cnns):
            conv = torch.nn.Conv1d(
                in_channels=self._word_dim,
                out_channels=self._n_filters,
                kernel_size=self._window,
                bias=True
            )
            conv.weight.requires_grad = self._require_grad
            conv.bias.requires_grad = self._require_grad
            # init conv
            init_cnn_weight(conv)
            self.add_module('lm_conv_{}'.format(i), conv)
            del conv

            # pad = get_padding(self._window, self._word_dim)
            # self._paddings['padding_{}'.format(i)] = pad
            # del pad

            # previous dataset
            # pad = WordFeature(self._window, self._word_dim, seed=2345)
            pad = WordFeature(self._window-1, self._word_dim, seed=2345)
            self.add_module('padding_{}'.format(i), pad)
            del pad

            reverse_conv = torch.nn.Conv1d(
                in_channels=self._word_dim,
                out_channels=self._n_filters,
                kernel_size=self._window,
                bias=True
            )
            reverse_conv.weight.requires_grad = self._require_grad
            reverse_conv.bias.requires_grad = self._require_grad
            # init conv
            init_cnn_weight(reverse_conv)
            self.add_module('lm_reverse_conv_{}'.format(i), reverse_conv)
            del reverse_conv

            # reverse_pad = get_padding(self._window, self._word_dim)
            # self._paddings['reverse_padding_{}'.format(i)] = reverse_pad
            # del reverse_pad

            # previous dataset
            # reverse_pad = WordFeature(self._window, self._word_dim, seed=2345)
            reverse_pad = WordFeature(self._window-1, self._word_dim, seed=2345)
            self.add_module('reverse_padding_{}'.format(i), reverse_pad)
            del reverse_pad

    def build_proj(self):
        proj = torch.nn.Linear(
            in_features=self._word_dim * 2,
            out_features=self._word_dim,
            bias=True
        )
        proj.weight.requires_grad = self._require_grad
        proj.bias.requires_grad = self._require_grad
        # init linear
        init_linear(proj)
        self.add_module('lm_proj', proj)
        del proj


    def forward(self, inputs):
        """
        Args:
            inputs: a tuple, include forward_inputs and backward_inputs
                    each input is a size of [batch_size, n_tokens, word_dim]
        padding, conv, proj, padding, conv, proj
        """
        batch_size, n_totkens = inputs[0].size()[0], inputs[0].size()[1]
        forward_inputs, backward_inputs = inputs
        del inputs
        # forward_inputs = forward_inputs[:, :-1, :]
        # backward_inputs = backward_inputs[:, :-1, :]

        for i in range(self._n_cnns):
            # previous dataset
            # forward_inputs = forward_inputs[:, :-1, :]
            # backward_inputs = backward_inputs[:, :-1, :]

            # forward cnn
            # paddings = self._paddings['padding_{}'.format(i)]
            # paddings = paddings.expand(batch_size, -1, -1)
            paddings = getattr(self, 'padding_{}'.format(i))._embed.weight
            paddings = paddings.expand(batch_size, -1, -1)

            forward_inputs = torch.cat((paddings, forward_inputs), dim=1)
            del paddings

            forward_inputs = torch.transpose(forward_inputs, 1, 2)
            # dropout layer
            forward_inputs = self._dropout(forward_inputs)

            # conv
            conv = getattr(self, 'lm_conv_{}'.format(i))
            convolved = conv(forward_inputs)

            # relu
            # convolved = F.relu(convolved)
            forward_inputs = torch.transpose(convolved, 1, 2)
            del conv
            del convolved

            # backward cnn
            #reverse_paddings = self._paddings['reverse_padding_{}'.format(i)]
            reverse_paddings = getattr(self, 'reverse_padding_{}'.format(i))._embed.weight
            reverse_paddings = reverse_paddings.expand(batch_size, -1, -1)

            backward_inputs = torch.cat((reverse_paddings, backward_inputs), dim=1)
            del reverse_paddings

            backward_inputs = torch.transpose(backward_inputs, 1, 2)
            backward_inputs = self._dropout(backward_inputs)

            reverse_conv = getattr(self, 'lm_reverse_conv_{}'.format(i))
            reverse_convolved = reverse_conv(backward_inputs)

            # reverse_convolved = F.relu(reverse_convolved)
            backward_inputs = torch.transpose(reverse_convolved, 1, 2)
            del reverse_conv
            del reverse_convolved

        # use forward context and backward context create the context
        # for each word
        idx = [i for i in range(backward_inputs.size(1) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        if self._gpu:
            idx = idx.cuda()
        reverse_backward_inputs = backward_inputs.index_select(1, idx)
        del backward_inputs
        del idx

        context = torch.cat((forward_inputs, reverse_backward_inputs), dim=2)
        del forward_inputs
        del reverse_backward_inputs

        if self._n_highway > 0:
            context = self._highway(context)

        # proj the context
        proj = getattr(self, 'lm_proj')
        context = proj(context)

        return context

