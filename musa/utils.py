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


def rmse(prediction, groundtruth):
    assert prediction.shape == groundtruth.shape
    D = (groundtruth - prediction) ** 2
    D = np.mean(D, axis=0)
    return np.sqrt(D)

def denorm_minmax(y, out_min, out_max):
    # x = y * (max - min) + min
    R = out_max - out_min
    return y * R + out_min
