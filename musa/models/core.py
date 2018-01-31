import torch
from torch.autograd import Variable
import torch.nn as nn
import os


class speaker_model(nn.Module):

    def __init__(self, num_inputs, mulspk_type, speakers=None, cuda=False):
        super(speaker_model, self).__init__()
        """
        # Arguments
            speakers: list of speakers to model. If None, just one speaker
                      is modeled.
            mulspk_type: type of multiple speaker representation: sinout
            (embedding layer), mulout (multiple outputs), gating (gated
            activations)
        """
        assert num_inputs > 0, num_inputs
        self.num_inputs = num_inputs
        self.do_cuda = cuda
        self.mulspk_type = mulspk_type
        self.mulout = (mulspk_type == 'mulout')
        self.sinout = (mulspk_type == 'sinout')
        self.gating = (mulspk_type == 'gating')
        if speakers is None or len(speakers) <= 1:
            self.speakers = None
            self.mulspk_type = 'sinout'
        else:
            self.speakers = speakers
            print('Modeling speakers: ', self.speakers)
            self.mulspk_type = mulspk_type

    def save(self, save_path, out_filename, epoch, best_val=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_w_fpath = 'e{}_{}.weights'.format(epoch, out_filename)
        out_fpath = 'e{}_{}'.format(epoch, out_filename)
        if best_val:
            out_fpath = 'best-val_{}'.format(out_fpath)
            out_w_fpath = 'best-val_{}'.format(out_w_fpath)
        out_fpath = os.path.join(save_path, out_fpath)
        out_w_fpath = os.path.join(save_path, out_w_fpath)
        torch.save(self.state_dict(), out_w_fpath)
        torch.save(self, out_fpath)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))


    def build_spk_embedding(self):
        self.input_fc = nn.Linear(self.num_inputs, self.emb_size)
        if self.speakers is not None:
            assert isinstance(self.speakers, list), type(self.speakers)
            if not self.mulout: 
                # prepare input embedding to distinguish b/w speakers
                self.emb = nn.Embedding(len(self.speakers), 
                                        len(self.speakers))
                if self.sinout:
                    # gating does not include embedding as input
                    self.emb_size = self.emb_size + len(self.speakers)
                print('emb_size: ', self.emb_size)

    def build_input_embedding(self):
        # two-layer tanh embedding projections
        if not self.gating:
            from collections import OrderedDict
            emb_fc_d = OrderedDict()
            emb_fc_d['act_0'] = getattr(nn, self.emb_activation)()
            if self.bnorm:
                emb_fc_d['bnorm_0'] = nn.BatchNorm1d(self.emb_size)
            for n in range(self.emb_layers - 1):
                emb_fc_d['fc_{}'.format(n + 1)] = nn.Linear(self.emb_size,
                                                             self.emb_size)
                emb_fc_d['dout_{}'.format(n + 1)] = nn.Dropout(self.dropout)
                emb_fc_d['act_{}'.format(n + 1)] = getattr(nn, self.emb_activation)()
                if self.bnorm:
                    emb_fc_d['bnorm_{}'.format(n + 1)] = nn.BatchNorm1d(self.emb_size)
            self.emb_fc = nn.Sequential(emb_fc_d)

            # first embedding layer is not fused at beginig
            #self.emb_fc = nn.Sequential(
            #    getattr(nn, self.emb_activation)(),
            #    nn.Linear(self.emb_size, self.emb_size),
            #    nn.Dropout(self.dropout),
            #    getattr(nn, self.emb_activation)(),
            #)
        else:
            # create gated trunk (from very beginning)
            self.emb_fc = [
                nn.Sequential(
                    nn.Linear(self.num_inputs, self.emb_size),
                    getattr(nn, self.emb_activation)(),
                ),
                nn.Sequential(
                    nn.Linear(self.emb_size, self.emb_size),
                    nn.Dropout(self.dropout),
                    getattr(nn, self.emb_activation)(),
                )
            ]
            self.emb_g = [
                nn.Sequential(
                    nn.Linear(self.num_inputs + len(self.speakers), 
                              self.emb_size),
                    nn.Sigmoid(),
                ),
                nn.Sequential(
                    nn.Linear(self.emb_size + len(self.speakers), self.emb_size),
                    self.Dropout(self.dropout),
                    self.Sigmoid()
                )
            ]

    def forward_input_embedding(self, dling_features, speaker_idx):
        # inputs are time-major (Seqlen, Bsize, features)
        re_dling_features = dling_features.view(-1, dling_features.size(-1))
        if not self.gating:
            # use the NOT gated mechanism (sinout, mulout)
            # go through fully connected embedding layers
            # if sinout, merge embedding vector for many 
            # speakers case
            in_h = self.input_fc(re_dling_features)
            in_h = in_h.view(dling_features.size(0), -1,
                             in_h.size(-1))
            if self.speakers is not None and self.sinout:
                # project speaker ID through embedding layer
                emb_h = self.emb(speaker_idx)
                # concat embeddings to first-layer activations
                in_h = torch.cat([in_h, emb_h], dim=-1)
            #print('in_h: ', in_h.size())
            #print('self.emb_size: ', self.emb_size)
            # forward through remaining fc embeddings
            x = self.emb_fc(in_h.view(-1, in_h.size(-1)))
        else:
            # use the gated mechanism
            x = re_dling_features
            for (tanh_l, sig_l) in zip(self.emb_fc,
                                       self.emb_g):
                x_tanh = tanh_l(x)
                x_sig = sig_l(x)
                x = x_tanh * x_sig
        x = x.view(dling_features.size(0), -1, 
                   self.emb_size)
        return x

    def build_core_rnn(self):
        # one-layer LSTM
        self.core_rnn = nn.LSTM(self.emb_size, self.rnn_size,
                                num_layers=self.rnn_layers,
                                bidirectional=False,
                                batch_first=False,
                                dropout=self.dropout)
        if self.gating:
            self.core_g = nn.Sequential(
                nn.Linear(self.emb_size, self.rnn_size),
                nn.Sigmoid()
            )

    def forward_core(self, inp, hid_state):
        h_t, hid_state = self.core_rnn(inp, hid_state)
        if self.gating:
            # we can forward 3-D tensors in linear now! [B, T, F]
            inp_g = inp.transpose(0, 1)
            x_g = self.core_g(inp_g)
            x_g = x_g.transpose(0, 1)
            assert x_g.size() == h_t.size(), x_g.size()
            h_t = x_g * h_t
        return h_t, hid_state

    def build_output(self, rnn_output=False):
        if self.mulout:
            # Multi-Output model
            # make as many out layers as speakers
            self.out_layers = {}
            for k in self.speakers:
                if rnn_output:
                    self.out_layers[k] = nn.LSTM(self.rnn_size,
                                                 self.num_outputs)
                else:
                    self.out_layers[k] = nn.Linear(self.rnn_size,
                                                   self.num_outputs)
                if self.do_cuda:
                    self.out_layers[k].cuda()
        else:
            if rnn_output:
                self.out_layer = nn.LSTM(self.rnn_size,
                                         self.num_outputs,
                                         num_layers=1,
                                         bidirectional=False,
                                         batch_first=False,
                                         dropout=0.)
            else:
                self.out_layer = nn.Linear(self.rnn_size,
                                           self.num_outputs)

