import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .core import speaker_model


class dur_discriminator(nn.Module):
    """ duration discriminator """

    def __init__(self, rnn_size, rnn_layers):
        super().__init__()
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        # -- Build recurrent component
        self.core_rnn = nn.LSTM(1, self.rnn_size,
                                num_layers=self.rnn_layers,
                                bidirectional=False,
                                batch_first=True,
                                dropout=0)
        # -- Build output mapping FC
        self.out_fc = nn.Linear(self.rnn_size, 1)
        
    def forward(self, dur_input, rnn_state=None):
        # inputs are time-minor (Bsize, Seqlen, features)
        x, rnn_state = self.core_rnn(dur_input, rnn_state)
        y = self.out_fc(rnn_state.view(self.rnn_layers,
                                       -1, self.rnn_size)
        return y, rnn_state

    def init_hidden_state(self, curr_bsz, volatile=False):
        return (Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size),
                         volatile=volatile),
                Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size), 
                         volatile=volatile))

