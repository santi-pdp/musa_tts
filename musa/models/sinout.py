import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .core import speaker_model


class acoustic_rnn(speaker_model):
    """ acoustic RNN model """

    def __init__(self, num_inputs, emb_size, rnn_size, rnn_layers,
                 dropout, sigmoid_out=True, speakers=None,
                 mulout=False, cuda=False):
        super(acoustic_rnn, self).__init__()
        """
        # Arguments
            speakers: list of speakers to model. If None, just one speaker
                      is modeled.
            mulout: specifies that every speaker is one output. This flag is
                    ignored if there are no more than 1 speaker. If False,
                    the multiple-speakers will be modeled with an input
                    embedding layer.
        """
        self.speakers = None
        assert num_inputs > 0, num_inputs
        self.emb_size = emb_size
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        #self.num_outputs = num_outputs
        self.num_outputs = 43
        self.dropout = dropout
        self.sigmoid_out = sigmoid_out
        if not sigmoid_out:
            print('BEWARE in aco model: not applying sigmoid to output, '
                  'you may obtain classification values out of binary range')
        self.num_inputs = num_inputs
        if speakers is None or len(speakers) <= 1:
            self.speakers = None
            self.mulout = False
        else:
            self.speakers = speakers
            self.mulout = mulout
        self.mulout = mulout
        self.num_inputs = num_inputs
        # -- Embedding layers (merge of input features)
        self.input_fc = nn.Linear(num_inputs, self.emb_size)
        if self.speakers is not None:
            assert type(speakers) == list, type(speakers)
            # list of speaker names
            self.speakers = speakers
            if not self.mulout:
                # prepare input embedding to distinguish b/w speakers
                self.emb = nn.Embedding(len(speakers), len(speakers))
                self.emb_size = self.emb_size + len(speakers)
                print('emb_size: ', self.emb_size)
        # two-layer tanh embedding projections
        self.emb_fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Dropout(self.dropout),
            nn.Tanh()
        )
        # -- Build recurrent component
        # one-layer LSTM
        self.core_rnn = nn.LSTM(self.emb_size, self.rnn_size,
                                num_layers=self.rnn_layers,
                                bidirectional=False,
                                batch_first=False,
                                dropout=self.dropout)
        # -- Build output mapping FC(s)
        if self.mulout:
            # Multi-Output model
            # make as many out layers as speakers
            self.out_rnn = {}
            for k in self.speakers:
                self.out_rnn[k] = nn.LSTM(self.rnn_size, self.num_outputs)
                if cuda:
                    self.out_rnn[k].cuda()
        else:
            # just one output layer
            self.out_rnn = nn.LSTM(self.rnn_size, self.num_outputs,
                                   num_layers=1, bidirectional=False,
                                   batch_first=False, dropout=0.)

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
        # inputs are time-major (Seqlen, Bsize, features)
        re_dling_features = dling_features.view(-1, dling_features.size(-1))
        # go through fully connected embedding layers
        in_h = self.input_fc(re_dling_features)
        in_h = in_h.view(dling_features.size(0), -1,
                         in_h.size(-1))
        if self.speakers is not None and not self.mulout:
            emb_h = self.emb(speaker_idx)
            #print('emb_h size: ', emb_h.size())
            # concat embeddings to first-layer activations
            in_h = torch.cat([in_h, emb_h], dim=-1)
        x = self.emb_fc(in_h.view(-1, in_h.size(-1)))
        x = x.view(dling_features.size(0), -1, 
                   self.emb_size)
        x, hid_state = self.core_rnn(x, hid_state)
        if self.mulout:
            y = {}
            nout_state = {}
            for spk in self.speakers:
                y[spk], \
                nout_state[spk] = self.out_rnn[spk](x,
                                                   out_state[spk])
                if self.sigmoid_out:
                    y[spk] = F.sigmoid(y[spk])
                y[spk] = y[spk].view(dling_features.size(0), -1,
                                     self.num_outputs)
        else:
            y, nout_state = self.out_rnn(x,
                                         out_state)
            if self.sigmoid_out:
                y = F.sigmoid(y)
            y = y.view(dling_features.size(0), -1, self.num_outputs)
        return y, hid_state, nout_state

    def init_hidden_state(self, curr_bsz, volatile=False):
        return (Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size),
                         volatile=volatile),
                Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size), 
                         volatile=volatile))

    def init_output_state(self, curr_bsz, volatile=False):
        if self.mulout:
            # return dict of output states, one per spk
            out_states = {}
            for spk in self.speakers:
                out_states[spk] = (Variable(torch.zeros(1, curr_bsz, 
                                                        self.num_outputs),
                                            volatile=volatile),
                                   Variable(torch.zeros(1, curr_bsz, 
                                                        self.num_outputs), 
                                            volatile=volatile))
        else:
            out_states = (Variable(torch.zeros(1, curr_bsz, 
                                               self.num_outputs),
                                   volatile=volatile),
                          Variable(torch.zeros(1, curr_bsz, 
                                               self.num_outputs), 
                                   volatile=volatile))
        return out_states

class sinout_duration(speaker_model):
    """ Baseline single output duration model """

    def __init__(self, num_inputs, num_outputs, emb_size, rnn_size, rnn_layers,
                 dropout, sigmoid_out=False, speakers=None, mulout=False,
                 cuda=False):
        super(sinout_duration, self).__init__()
        """
        # Arguments
            speakers: list of speakers to model. If None, just one speaker
                      is modeled.
            mulout: specifies that every speaker is one output. This flag is
                    ignored if there are no more than 1 speaker. If False,
                    the multiple-speakers will be modeled with an input
                    embedding layer.
        """
        self.speakers = None
        assert num_inputs > 0, num_inputs
        self.emb_size = emb_size
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.sigmoid_out = sigmoid_out
        self.num_inputs = num_inputs
        if speakers is None or len(speakers) <= 1:
            self.speakers = None
            self.mulout = False
        else:
            self.speakers = speakers
            self.mulout = mulout
        # -- Embedding layers (merge of input features)
        self.input_fc = nn.Linear(num_inputs, self.emb_size)
        if self.speakers is not None:
            assert type(speakers) == list, type(speakers)
            # list of speaker names
            self.speakers = speakers
            if not self.mulout:
                # prepare input embedding to distinguish b/w speakers
                self.emb = nn.Embedding(len(speakers), len(speakers))
                self.emb_size = self.emb_size + len(speakers)
                print('emb_size: ', self.emb_size)
        # two-layer tanh embedding projections
        self.emb_fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Dropout(self.dropout),
            nn.Tanh()
        )
        # -- Build recurrent component
        # one-layer LSTM
        self.core_rnn = nn.LSTM(self.emb_size, self.rnn_size,
                                num_layers=self.rnn_layers,
                                bidirectional=False,
                                batch_first=False,
                                dropout=self.dropout)
        # -- Build output mapping FC(s)
        if self.mulout:
            # Multi-Output model
            # make as many out layers as speakers
            self.out_fc = {}
            for k in self.speakers:
                self.out_fc[k] = nn.Linear(self.rnn_size, self.num_outputs)
                if cuda:
                    self.out_fc[k].cuda()
        else:
            # just one output
            self.out_fc = nn.Linear(self.rnn_size, self.num_outputs)
        if self.num_outputs > 1:
            self.out_g = nn.LogSoftmax()
        

    def forward(self, ling_features, rnn_state=None, speaker_idx=None):
        """ Forward the linguistic features, and the speaker ID
            # Arguments
                ling_features: Tensor with encoded linguistic features
                rnn_states: dict containing RNN layers' hidden states
                speaker_id: Tensor with speaker idx to be generated
                In case of MO model, all speakers are generated, so
                speaker_idx is not needed.
        """
        # inputs are time-major (Seqlen, Bsize, features)
        re_ling_features = ling_features.view(-1, ling_features.size(-1))
        # go through fully connected embedding layers
        in_h = self.input_fc(re_ling_features)
        in_h = in_h.view(ling_features.size(0), -1,
                         in_h.size(-1))
        if self.speakers is not None and not self.mulout:
            emb_h = self.emb(speaker_idx)
            #print('emb_h size: ', emb_h.size())
            # concat embeddings to first-layer activations
            in_h = torch.cat([in_h, emb_h], dim=-1)
        x = self.emb_fc(in_h.view(-1, in_h.size(-1)))
        x = x.view(ling_features.size(0), -1, 
                   self.emb_size)
        x, rnn_state = self.core_rnn(x, rnn_state)
        if self.mulout:
            y = {}
            for spk in self.speakers:
                y[spk] = self.out_fc[spk](x.view(-1, self.rnn_size))
                if self.sigmoid_out and self.num_outputs == 1:
                    y[spk] = F.sigmoid(y[spk])
                elif self.num_outputs > 1:
                    y[spk] = self.out_g(y[spk])
                y[spk] = y[spk].view(ling_features.size(0), -1,
                                     self.num_outputs)
        else:
            y = self.out_fc(x.view(-1, self.rnn_size))
            if self.sigmoid_out and self.num_outputs == 1:
                y = F.sigmoid(y)
            elif self.num_outputs > 1:
                y = self.out_g(y)
            y = y.view(ling_features.size(0), -1, self.num_outputs)
        return y, rnn_state

    def init_hidden_state(self, curr_bsz, volatile=False):
        return (Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size),
                         volatile=volatile),
                Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size), 
                         volatile=volatile))

