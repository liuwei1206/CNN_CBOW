__author__ = "liuwei"

"""
the model to train w2v use only CNN
"""
import torch
import numpy as np
import torch.nn as nn

import os
import re
import logging

class CNN_W2V_Model(torch.nn.Module):
    def __init__(self, embed, cnn_lm, loss, full_loss, gpu):
        """
        Args:
            embed: convert data ids to embedding data
            cnn_lm: cnn language model
            loss: loss function
            gpu:
        """
        super(CNN_W2V_Model, self).__init__()

        self._embed = embed
        self._cnn_lm = cnn_lm
        self._loss = loss
        self._full_loss = full_loss
        self._gpu = gpu

    def forward(self, inputs, validation=False):
        """
        Args:
            inputs: a dict of inputs data, it is a dict of Tensor
            {
                'batch_word_ids': batch_word_ids,
                'reverse_batch_word_ids': reverse_batch_word_ids
            }
            validation: if True, use full_loss, else use loss
        """
        batch = inputs['batch_word_ids']
        reverse_batch = inputs['reverse_batch_word_ids']
        target = inputs['batch_target_ids']

        del inputs

        if self._gpu:
            batch = batch.cuda()
            reverse_batch = reverse_batch.cuda()

        # [batch_size, n_tokens, word_dim]
        batch_embedding_inputs = self._embed(batch)
        reverse_batch_embedding_inputs = self._embed(reverse_batch)

        del reverse_batch
        del batch

        # cnn_lm
        context = self._cnn_lm((batch_embedding_inputs,
                                reverse_batch_embedding_inputs))
        del batch_embedding_inputs
        del reverse_batch_embedding_inputs
        
        if self._gpu:
            target = target.cuda()
        # cal loss
        # if validation:
        #      loss, accu = self._full_loss(context, batch)
        # else:
        #     loss, accu = self._loss(context, batch)
        if validation:
            loss, accu = self._full_loss(context, target)
        else:
            loss, accu = self._loss(context, target)

        # loss, accu = self._loss(context, batch)
        # print('out')

        del context
        # del batch
        del target

        return (loss, accu)
