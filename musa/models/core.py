import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import copy
import math
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

    def describe_model(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if
                                   p.requires_grad)
        print('Total params: ', pytorch_total_params)

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
            if self.emb_activation != 'None':
                emb_fc_d['act_0'] = getattr(nn, self.emb_activation)()
            if hasattr(self, 'bnorm') and self.bnorm:
                emb_fc_d['bnorm_0'] = nn.BatchNorm1d(self.emb_size)
            for n in range(self.emb_layers - 1):
                emb_fc_d['fc_{}'.format(n + 1)] = nn.Linear(self.emb_size,
                                                             self.emb_size)
                emb_fc_d['dout_{}'.format(n + 1)] = nn.Dropout(self.dropout)
                if self.emb_activation != 'None':
                    emb_fc_d['act_{}'.format(n + 1)] = getattr(nn, self.emb_activation)()
                if hasattr(self, 'bnorm') and self.bnorm:
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
        dling_features = dling_features.transpose(0, 1)
        #re_dling_features = dling_features.view(-1, dling_features.size(-1))
        #re_dling_features = dling_features.view(-1, dling_features.size(-1))
        if not self.gating:
            # use the NOT gated mechanism (sinout, mulout)
            # go through fully connected embedding layers
            # if sinout, merge embedding vector for many 
            # speakers case
            in_h = self.input_fc(dling_features)
            if self.speakers is not None and self.sinout:
                # project speaker ID through embedding layer
                emb_h = self.emb(speaker_idx)
                # concat embeddings to first-layer activations
                in_h = torch.cat([in_h, emb_h], dim=-1)
            #print('in_h: ', in_h.size())
            #print('self.emb_size: ', self.emb_size)
            # forward through remaining fc embeddings
            x = self.emb_fc(in_h)
        else:
            # use the gated mechanism
            x = dling_features
            for (tanh_l, sig_l) in zip(self.emb_fc,
                                       self.emb_g):
                x_tanh = tanh_l(x)
                x_sig = sig_l(x)
                x = x_tanh * x_sig
        #x = x.view(dling_features.size(0), -1, 
        #           self.emb_size)
        x = x.transpose(0, 1)
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
                                                 self.num_outputs,
                                                 num_layers=1,
                                                 bidirectional=False,
                                                 batch_first=False,
                                                 dropout=0.)
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

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class AttEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AttEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class AttDecoderLayer(nn.Module):
    "Decoder is made up of self-attn, src-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(AttEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = serc_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, mask))
        return self.sublayer[2](x, self.feed_forward)

class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(model):
    return NoamOpt(model.emb_size, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
