__author__ = "liuwei"

"""
padding is also a tainable variable, different direction ,
different layer use diffrent padding
"""

import torch
import numpy as np
from torch.autograd import Variable
from .iniatlize import init_embedding

def get_padding(pad_len, pad_dim, require_grad=True):
    """
    Args:
        pad_len: pad how much
        pad_dim: size of padding dim
    """
    input_paddings = torch.FloatTensor(
        size=(pad_len, pad_dim)
    )
    input_paddings.requires_grad = require_grad
    # init the padding
    init_embedding(input_paddings)

    input_paddings = torch.autograd.Variable(
        input_paddings, requires_grad=require_grad
    )

    return input_paddings