import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F
from .core import *
import copy


def acoustic_builder(aco_type, opts):
    if aco_type == 'rnn':
        model = acoustic_rnn(num_inputs=opts.num_inputs, 
                             emb_size=opts.emb_size,
                             rnn_size=opts.rnn_size,
                             rnn_layers=opts.rnn_layers,
                             dropout=opts.dout,
                             speakers=opts.spks,
                             mulout=opts.mulout,
                             cuda=opts.cuda,
                             emb_layers=opts.emb_layers,
                             emb_activation=opts.emb_activation)
        train_fn = 'train_aco_epoch'
        eval_fn = 'eval_aco_epoch'
    elif aco_type == 'satt':

        model = acoustic_satt(num_inputs=opts.num_inputs,
                             emb_size=opts.emb_size,
                             dropout=opts.dout,
                             emb_activation=opts.emb_activation,
                             speakers=opts.spks,
                             emb_layers=opts.emb_layers,
                             d_ff=opts.d_ff,
                             cuda=opts.cuda,
                             N=opts.N,
                             h=opts.h,
                             lnorm=(not opts.no_lnorm),
                             conv_out=opts.conv_out)
        train_fn = 'train_attaco_epoch'
        eval_fn = 'eval_attaco_epoch'
    elif aco_type == 'decsatt':
        model = acoustic_decoder_satt(num_inputs=opts.num_inputs,
                                      emb_size=opts.emb_size,
                                      dropout=opts.dout,
                                        cuda=opts.cuda,
                                      emb_activation=opts.emb_activation,
                                      speakers=opts.spks,
                                      emb_layers=opts.emb_layers,
                                      d_ff=opts.d_ff,
                                      N=opts.N,
                                      h=opts.h)
        train_fn = 'train_attaco_epoch'
        eval_fn = 'eval_attaco_epoch'
    else:
        raise TypeError('Unrecognized model type: ', aco_type)
    return model, train_fn, eval_fn

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
                #y[spk] = self.correct_classification_output(y[spk])
                y[spk] = tanh2sigmoid(y[spk])
                #y[spk] = y[spk].view(dling_features.size(0), -1,
                #                     self.num_outputs)
        else:
            y, nout_state = self.out_layer(x,
                                           out_state)
            # Bound classification output within [0, 1] properly
            #y = correct_classification_output(y)
            y = tanh2sigmoid(y)
            #y = y.view(dling_features.size(0), -1, self.num_outputs)
        return y, hid_state, nout_state


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
                 d_ff=2048, N=6,
                 out_activation='Sigmoid',
                 lnorm=True,
                 conv_out=False):
        # no other mulspk implemented yet
        assert mulspk_type == 'sinout', mulspk_type
        super().__init__(num_inputs, mulspk_type, 
                         speakers=speakers,
                         cuda=cuda)
        self.emb_size = emb_size
        self.emb_layers = emb_layers
        self.emb_activation = emb_activation
        self.bnorm = bnorm
        self.num_outputs = 43
        self.dropout = dropout
        self.num_inputs = num_inputs
        # build embedding of spks (if required) 
        self.build_spk_embedding()
        # wrongly named rnn_size, there are no RNN here
        # but core needs this name for output layer
        self.rnn_size = self.emb_size
        # -- Build tanh embedding trunk
        self.build_input_embedding()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, self.emb_size, dropout=dropout)
        ff = PositionwiseFeedForward(self.emb_size, d_ff, dropout)
        self.position = PositionalEncoding(self.emb_size, dropout)
        enc_layer = AttEncoderLayer(self.emb_size, c(attn), c(ff),
                                    dropout, lnorm)
        self.model = clones(enc_layer, N)
        if lnorm:
            self.norm = LayerNorm(enc_layer.size)
        # -- Build output mapping FC(s)
        if conv_out:
            self.out_layer = nn.Conv1d(emb_size, self.num_outputs, 21,
                                       padding=10)
        else:
            self.build_output(rnn_output=False)
        self.conv_out = conv_out
        self.out_activation = out_activation
        self.sigmoid = nn.Sigmoid()
        #print('Built enc_layer: ', enc_layer)
        #print('Built norm layer: ', self.norm)
        #print('Built model: ', self.model)
        print(self)
        print('Initializing xavier weights...')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dling_features, 
                speaker_idx=None,
                pe_start_idx=0):
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
            x = self.position(x, pe_start_idx)
        #print('x size: ', x.size())
        h = x
        # Now we will forward through the transformer encoder structure
        for layer in self.model:
            h = layer(h, None)
        #print('h size: ', h.size())
        if hasattr(self, 'norm'):
            h = self.norm(h)
        # forward through output FC
        if self.conv_out:
            h = h.transpose(1, 2)
            y = self.out_layer(h)
            y = y.transpose(1, 2)
        else:
            y = self.out_layer(h)
        y = self.sigmoid(y)
        y = y.transpose(0, 1)
        return y

class acoustic_decoder_satt(speaker_model):
    # TODO: Check validity of this model in terms of seq2seq behavior
    def __init__(self, num_inputs, emb_size=512, 
                 dropout=0.1,
                 emb_activation='Tanh',
                 speakers=None,
                 mulspk_type='sinout',
                 mulout=False, cuda=False,
                 bnorm=False,
                 emb_layers=2,
                 h=8, d_model=512,
                 d_ff=2048, N=6,
                 out_activation='Sigmoid'):
        # no other mulspk implemented yet
        assert mulspk_type == 'sinout', mulspk_type
        super().__init__(num_inputs, mulspk_type, 
                         speakers=speakers,
                         cuda=cuda)
        self.do_cuda = cuda
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
        self.aco_W = nn.Linear(43, emb_size)
        enc_layer = AttDecoderLayer(emb_size, c(attn), c(attn),
                                    c(ff),
                                    dropout)
        self.model = clones(enc_layer, N)
        self.norm = LayerNorm(enc_layer.size)
        # -- Build output mapping FC(s)
        self.build_output(rnn_output=False)
        self.sigmoid = getattr(nn, opts.out_activation)()
        #print('Built enc_layer: ', enc_layer)
        #print('Built norm layer: ', self.norm)
        #print('Built model: ', self.model)
        print(self)
        print('Initializing xavier weights...')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dling_features, 
                prev_aco_features=None,
                speaker_idx=None, 
                pe_start_idx=0):
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
        if prev_aco_features is None:
            # build zero tensor and begin loop
            bos_token = torch.zeros(dling_features.size(1),
                                    1,
                                    43)
            if self.do_cuda:
                bos_token = bos_token.cuda()
            aco_tokens = []
            aco_token = bos_token
            for aco_i in range(dling_features.size(0)):
                #print('Forwarding {}/{} aco frame'.format(aco_i, 
                #                                          dling_features.size(0)))
                aco = self.aco_W(aco_token)
                if hasattr(self, 'position'):
                    aco = self.position(aco, pe_start_idx)
                m = x
                h = aco
                # Now we will forward through the transformer encoder structure
                for layer in self.model:
                    h = layer(h, m, None, None)
                h = self.norm(h)
                # forward through output FC
                y = self.out_layer(h)
                y = self.sigmoid(y)
                aco_token = y
                aco_tokens.append(aco_token)
            aco_tokens = torch.cat(aco_tokens, dim=1)
            return aco_tokens.transpose(0, 1)
        else: 
            aco = self.aco_W(prev_aco_features.transpose(0,1))
            if hasattr(self, 'position'):
                aco = self.position(aco)
            #print('x size: ', x.size())
            m = x
            h = aco
            mask = subsequent_mask(h.size(1))
            if self.do_cuda:
                mask = mask.cuda()
            # Now we will forward through the transformer encoder structure
            for layer in self.model:
                h = layer(h, m, None, mask)
            #print('h size: ', h.size())
            h = self.norm(h)
            # forward through output FC
            y = self.out_layer(h)
            y = self.sigmoid(y)
            y = y.transpose(0, 1)
            return y
