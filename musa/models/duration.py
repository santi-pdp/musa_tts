import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .core import speaker_model


class duration_rnn(speaker_model):
    """ Baseline duration model with RNN """

    def __init__(self, num_inputs, num_outputs, emb_size, rnn_size, rnn_layers,
                 dropout, sigmoid_out=False, speakers=None, mulout=False,
                 cuda=False, emb_layers=1, emb_act='PReLU'):
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
        # TODO: delete option to specify more outputs
        assert num_outputs == 1, num_outputs
        assert num_inputs > 0, num_inputs
        self.emb_size = emb_size
        self.emb_layers = emb_layers
        # emb_activation and stuff will be processed
        # in super class
        self.emb_activation = emb_act
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
        # ----------------------------------------------
        # build embedding of spks (if required) 
        self.build_spk_embedding()
        # -- Build tanh embedding trunk
        self.build_input_embedding()
        # -- Build recurrent component
        self.build_core_rnn()
        # -- Build output mapping FC(s)
        self.build_output(rnn_output=False)
        if sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, ling_features, rnn_state=None, speaker_idx=None):
        """ Forward the linguistic features, and the speaker ID
            # Arguments
                ling_features: Tensor with encoded linguistic features
                rnn_states: dict containing RNN layers' hidden states
                speaker_id: Tensor with speaker idx to be generated
                In case of MO model, all speakers are generated, so
                speaker_idx is not needed.
        """
        x = self.forward_input_embedding(ling_features, speaker_idx)
        x, rnn_state = self.forward_core(x, rnn_state)
        if self.mulout:
            y = {}
            for spk in self.speakers:
                x = x.transpose(0, 1)
                y[spk] = self.out_layers[spk](x)
                if self.sigmoid_out and self.num_outputs == 1:
                    y[spk] = self.sigmoid(y[spk])
                y[spk] = y[spk].transpose(0, 1)
        else:
            x = x.transpose(0, 1)
            y = self.out_layer(x)
            if self.sigmoid_out and self.num_outputs == 1:
                y = self.sigmoid(y)
            y = y.transpose(0, 1)
        return y, rnn_state

    def init_hidden_state(self, curr_bsz):
        return (torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size),
                torch.zeros(self.rnn_layers, curr_bsz, self.rnn_size))

