import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .core import speaker_model


class duration_rnn(speaker_model):
    """ Baseline single output duration model """

    def __init__(self, num_inputs, num_outputs, emb_size, rnn_size, rnn_layers,
                 dropout, sigmoid_out=False, speakers=None, mulout=False,
                 cuda=False, emb_layers=1):
        if mulout:
            mulspk_type = 'mulout'
        else:
            mulspk_type = 'sinout'
        super().__init__(num_inputs, mulspk_type, 
                         speakers=speakers,
                         cuda=cuda)
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
        self.emb_layers = emb_layers
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
        print('input_fc dur num_inputs: ', num_inputs)
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
        # TODO: update output building
        #self.build_output(rnn_output=False)
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

