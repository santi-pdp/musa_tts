import torch
from torch.autograd import Variable
import torch.nn as nn
import os


class speaker_model(nn.Module):

    def __init__(self):
        super(speaker_model, self).__init__()

    def save(self, save_path, out_filename, epoch, best_val=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_fpath = 'e{}_{}'.format(epoch, out_filename)
        if best_val:
            out_fpath = 'best-val_{}'.format(out_fpath)
        out_fpath = os.path.join(save_path, out_fpath)
        torch.save(self.state_dict(), out_fpath)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

