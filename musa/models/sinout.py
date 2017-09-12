import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .core import speaker_model


class sinout_acoustic(speaker_model):
    """ Baseline single output acoustic model """

    def __init__(self, num_inputs, speakers=None):
        super(sinout_acoustic, self).__init__()
        """
        # Arguments
            speakers: list of speakers to model. If None, just one speaker
                      is modeled.
        """
        self.speakers = None
        assert num_inputs > 0, num_inputs
        self.emb_size = 256
        self.rnn_size = 512
        self.num_outputs = 43
        self.num_inputs = num_inputs
        # -- Embedding layers (merge of input features)
        self.input_fc = nn.Linear(num_inputs, self.emb_size)
        if speakers is not None:
            assert type(speakers) == list, type(speakers)
            if len(speakers) <= 1:
                raise ValueError('If you specify a speaker list, you need at '
                                 'least > 2')
            # list of speaker names
            self.speakers = speakers
            self.emb = nn.Embedding(len(speakers), len(speakers))
            self.emb_size = self.emb_size + len(speakers)
        # two-layer tanh embedding projections
        self.emb_fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Tanh()
        )
        # -- Build recurrent component
        # one-layer LSTM
        self.core_rnn = nn.LSTM(self.emb_size, self.rnn_size)
        # -- Build output mapping RNN
        self.out_rnn = nn.LSTM(self.rnn_size, self.num_outputs)

    def forward(self, dling_features, hid_state, out_state, speaker_idx=None):
        """ Forward the duration + linguistic features, and the speaker ID
            # Arguments
                dling_features: Tensor with encoded linguistic features and
                duration (absolute + relative)
                speaker_id: Tensor with speaker idx to be generated
        """
        # inputs are time-major (Seqlen, Bsize, features)
        re_dling_features = dling_features.view(-1, dling_features.size(-1))
        in_h = self.input_fc(re_dling_features)
        in_h = in_h.view(dling_features.size(0), -1,
                         in_h.size(-1))
        if self.speakers is not None:
            emb_h = self.emb(speaker_idx)
            print('emb_h: ', emb_h.size())
            # concat embeddings to first-layer activations
            in_h = torch.cat([in_h, emb_h], dim=-1)
        x = self.emb_fc(in_h.view(-1, in_h.size(-1)))
        x = x.view(dling_features.size(0), -1, 
                   self.emb_size)
        x, hid_state = self.core_rnn(x, hid_state)
        y, out_state = self.out_rnn(x, out_state)
        return y, hid_state, out_state

    def init_hidden_state(self, curr_bsz):
        return (Variable(torch.zeros(1, curr_bsz, self.rnn_size)),
                Variable(torch.zeros(1, curr_bsz, self.rnn_size)))

    def init_output_state(self, curr_bsz):
        return (Variable(torch.zeros(1, curr_bsz, self.num_outputs)),
                Variable(torch.zeros(1, curr_bsz, self.num_outputs)))

class sinout_duration(speaker_model):
    """ Baseline single output duration model """

    def __init__(self, num_inputs, emb_size, rnn_size, rnn_layers,
                 dropout, sigmoid_out=False, speakers=None):
        super(sinout_duration, self).__init__()
        """
        # Arguments
            speakers: list of speakers to model. If None, just one speaker
                      is modeled.
        """
        self.speakers = None
        assert num_inputs > 0, num_inputs
        self.emb_size = emb_size
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.num_outputs = 1
        self.dropout = dropout
        self.sigmoid_out = sigmoid_out
        self.num_inputs = num_inputs
        # -- Embedding layers (merge of input features)
        self.input_fc = nn.Linear(num_inputs, self.emb_size)
        if speakers is not None:
            assert type(speakers) == list, type(speakers)
            if len(speakers) <= 1:
                raise ValueError('If you specify a speaker list, you need at '
                                 'least > 2')
            # list of speaker names
            self.speakers = speakers
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
        # -- Build output mapping FC
        self.out_fc = nn.Linear(self.rnn_size, self.num_outputs)
        

    def forward(self, ling_features, rnn_state, speaker_idx=None):
        """ Forward the linguistic features, and the speaker ID
            # Arguments
                ling_features: Tensor with encoded linguistic features
                rnn_states: dict containing RNN layers' hidden states
                speaker_id: Tensor with speaker idx to be generated
        """
        # inputs are time-major (Seqlen, Bsize, features)
        re_ling_features = ling_features.view(-1, ling_features.size(-1))
        # go through fully connected embedding layers
        in_h = self.input_fc(re_ling_features)
        in_h = in_h.view(ling_features.size(0), -1,
                         in_h.size(-1))
        if self.speakers is not None:
            emb_h = self.emb(speaker_idx)
            # concat embeddings to first-layer activations
            in_h = torch.cat([in_h, emb_h], dim=-1)
        x = self.emb_fc(in_h.view(-1, in_h.size(-1)))
        x = x.view(ling_features.size(0), -1, 
                   self.emb_size)
        x, rnn_state = self.core_rnn(x, rnn_state)
        y = self.out_fc(x.view(-1, self.rnn_size))
        if self.sigmoid_out:
            y = F.sigmoid(y)
        y = y.view(ling_features.size(0), -1, 1)
        return y, rnn_state

    def init_hidden_state(self, curr_bsz, volatile=False):
        return (Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size),
                         volatile=volatile),
                Variable(torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size), 
                         volatile=volatile))
