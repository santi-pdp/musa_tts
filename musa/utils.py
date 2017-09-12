import torch
from torch.autograd import Variable
import numpy as np


def var_to_cuda(var):
    if isinstance(var, Variable):
        return var.cuda()
    elif isinstance(var, tuple): 
        return tuple(v.cuda() for v in var)
    elif isinstance(var, list):
        return [v.cuda() for v in var]
    else:
        raise TypeError('Incorrect var type to cuda')

def repackage_hidden(h, curr_bsz):
    """ Coming from https://github.com/pytorch/examples/blob/master/word_language_model/main.py """
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data[:, :curr_bsz, :]).contiguous()
    else:
        return tuple(repackage_hidden(v, curr_bsz).contiguous() for v in h)


def rmse(prediction, groundtruth):
    assert prediction.shape == groundtruth.shape
    D = np.sqrt(np.mean((groundtruth - prediction) ** 2, axis=0))
    return D

def denorm_minmax(y, out_min, out_max):
    # x = y * (max - min) + min
    R = out_max - out_min
    return y * R + out_min
