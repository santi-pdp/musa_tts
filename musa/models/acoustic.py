import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F
from .core import *
import copy


def acoustic_builder(aco_type, opts):
    if aco_type == 'rnn':
        return acoustic_rnn(num_inputs=opts.num_inputs,
                            emb_size=opts.emb_size,
                            rnn_size=opts.rnn_size,
                            rnn_layers=opts.rnn_layers,
                            dropout=opts.dout,
                            speakers=opts.spks,
                            mulout=opts.mulout,
                            cuda=opts.cuda,
                            emb_layers=opts.emb_layers,
                            emb_activation=opts.emb_activation)
    elif aco_type == 'satt':
        return acoustic_satt(num_inputs=opts.num_inputs,
                             emb_size=opts.emb_size,
                             dropout=opts.dout,
                             emb_activation=opts.emb_activation,
                             speakers=opts.spks,
                             emb_layers=opts.emb_layers,
                             d_ff=opts.d_ff,
                             N=opts.N,
                             h=opts.h)
    else:
        raise TypeError('Unrecognized model type: ', aco_type)

class acoustic_rnn(speaker_model):
    """ acoustic RNN model """

    def __init__(self, num_inputs, emb_size, 
                 rnn_size, rnn_layers,
                 dropout, emb_activation='Tanh',
                 speakers=None,
                 mulspk_type='sinout',
                 mulout=False, cuda=False,
                 bnorm=False,
                 emb_layers=2):
        super().__init__(num_inputs, mulspk_type, 
                         speakers=speakers,
                         cuda=cuda)
        self.emb_size = emb_size
        self.emb_layers = emb_layers
        self.emb_activation = emb_activation
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.bnorm = bnorm
        self.num_outputs = 43
        self.dropout = dropout
        self.num_inputs = num_inputs
        print('aco_rnn num_inputs=', num_inputs)
        # build embedding of spks (if required) 
        self.build_spk_embedding()
        # -- Build tanh embedding trunk
        self.build_input_embedding()
        # -- Build recurrent component
        self.build_core_rnn()
        # -- Build output mapping RNN(s)
        self.build_output(rnn_output=True)

    def forward(self, dling_features, hid_state=None, out_state=None,
                speaker_idx=None):
        """ Forward the duration + linguistic features, and the speaker ID
            # Arguments
                dling_features: Tensor with encoded linguistic features and
                duration (absolute + relative)
                speaker_id: Tensor with speaker idx to be generated
        """
        if self.mulout and out_state is not None:
            assert isinstance(out_state, dict), type(out_state)
        if self.mulout and out_state is None:
            out_state = dict((spk, None) for spk in self.speakers)
        # forward through embedding
        x = self.forward_input_embedding(dling_features, speaker_idx)
        # forward through RNN core
        x, hid_state = self.forward_core(x, hid_state)
        # forward through output RNN
        if self.mulout:
            y = {}
            nout_state = {}
            for spk in self.speakers:
                y[spk], \
                nout_state[spk] = self.out_layers[spk](x,
                                                       out_state[spk])
                # Bound classification output within [0, 1] properly
                y[spk] = self.correct_classification_output(y[spk])
                #y[spk] = y[spk].view(dling_features.size(0), -1,
                #                     self.num_outputs)
        else:
            y, nout_state = self.out_layer(x,
                                           out_state)
            # Bound classification output within [0, 1] properly
            y = self.correct_classification_output(y)
            #y = y.view(dling_features.size(0), -1, self.num_outputs)
        return y, hid_state, nout_state

    def correct_classification_output(self, x):
        # split tensor in two, with last index being the one to go [0, 1]
        lin = x[:, :, :-1]
        cla = x[:, :, -1:]
        # suposing it went through TANH
        cla = (1 + cla) / 2
        return torch.cat((lin, cla), dim=2)

    def init_hidden_state(self, curr_bsz):
        return (torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size),
                torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size))

    def init_output_state(self, curr_bsz):
        if self.mulout:
            # return dict of output states, one per spk
            out_states = {}
            for spk in self.speakers:
                out_states[spk] = (torch.zeros(1, curr_bsz, 
                                               self.num_outputs),
                                   torch.zeros(1, curr_bsz, 
                                               self.num_outputs))
        else:
            out_states = (torch.zeros(1, curr_bsz, 
                                      self.num_outputs),
                          torch.zeros(1, curr_bsz, 
                                      self.num_outputs))
        return out_states


class acoustic_satt(speaker_model):

    def __init__(self, num_inputs, emb_size=512, 
                 dropout=0.1,
                 emb_activation='Tanh',
                 speakers=None,
                 mulspk_type='sinout',
                 mulout=False, cuda=False,
                 bnorm=False,
                 emb_layers=2,
                 h=8, d_model=512,
                 d_ff=2048, N=6):
        # no other mulspk implemented yet
        assert mulspk_type == 'sinout', mulspk_type
        super().__init__(num_inputs, mulspk_type, 
                         speakers=speakers,
                         cuda=cuda)
        self.emb_size = emb_size
        self.emb_layers = emb_layers
        self.emb_activation = emb_activation
        # wrongly named rnn_size, there are no RNN here
        # but core needs this name for output layer
        self.rnn_size = emb_size
        self.bnorm = bnorm
        self.num_outputs = 43
        self.dropout = dropout
        self.num_inputs = num_inputs
        # build embedding of spks (if required) 
        self.build_spk_embedding()
        # -- Build tanh embedding trunk
        self.build_input_embedding()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, emb_size)
        ff = PositionwiseFeedForward(emb_size, d_ff, dropout)
        self.position = PositionalEncoding(emb_size, dropout)
        enc_layer = AttEncoderLayer(emb_size, c(attn), c(ff),
                                    dropout)
        self.model = clones(enc_layer, N)
        self.norm = LayerNorm(enc_layer.size)
        # -- Build output mapping FC(s)
        self.build_output(rnn_output=False)
        self.sigmoid = nn.Sigmoid()
        #print('Built enc_layer: ', enc_layer)
        #print('Built norm layer: ', self.norm)
        #print('Built model: ', self.model)
        print(self)
        print('Initializing xavier weights...')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dling_features, hid_state=None, out_state=None,
                speaker_idx=None):
        """ Forward the duration + linguistic features, and the speaker ID
            # Arguments
                dling_features: Tensor with encoded linguistic features and
                duration (absolute + relative)
                speaker_id: Tensor with speaker idx to be generated
        """
        # states are ignored, nothing useful is carried there
        if self.mulout and out_state is not None:
            assert isinstance(out_state, dict), type(out_state)
        if self.mulout and out_state is None:
            out_state = dict((spk, None) for spk in self.speakers)
        # forward through embedding
        x = self.forward_input_embedding(dling_features, speaker_idx)
        x = x.transpose(0, 1)
        if hasattr(self, 'position'):
            x = self.position(x)
        #print('x size: ', x.size())
        h = x
        # Now we will forward through the transformer encoder structure
        for layer in self.model:
            h = layer(h, None)
        #print('h size: ', h.size())
        h = self.norm(h)
        # forward through output FC
        if self.mulout:
            y = {}
            for spk in self.speakers:
                y[spk] = self.out_layers[spk](x)
                y[spk] = self.sigmoid(y[spk])
                y[spk] = y[spk].transpose(0, 1)
        else:
            y = self.out_layer(x)
            y = self.sigmoid(y)
            y = y.transpose(0, 1)
        return y, None, None

    def init_hidden_state(self, curr_bsz):
        return None

    def init_output_state(self, curr_bsz):
        return None

