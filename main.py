import argparse
from musa.datasets import *
from musa.models import *
from torch.utils.data import DataLoader
import torch.optim as optim
from musa.core import *
import json
import os


def main(opts):
    with open(os.path.join(opts.save_path, 
                           'main.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    if opts.train_dur:
        # build dataset and loader
        dset = TCSTAR_dur(opts.cfg_spk, 'train',
                          opts.lab_dir, opts.codebooks_dir,
                          force_gen=opts.force_gen,
                          norm_dur=True,
                          max_spk_samples=opts.dur_max_samples,
                          parse_workers=opts.parser_workers,
                          max_seq_len=opts.dur_max_seq_len,
                          batch_size=opts.batch_size,
                          q_classes=opts.dur_q_classes)
        spk2durstats = dset.spk2durstats
        try:
            print('Spk2durstats: ', json.dumps(spk2durstats, indent=2))
        except TypeError as e:
            print(e)
        dloader = DataLoader(dset, batch_size=opts.batch_size, shuffle=True,
                             num_workers=opts.loader_workers, 
                             collate_fn=varlen_dur_collate)
        # build validation dataset and loader
        val_dset = TCSTAR_dur(opts.cfg_spk, 'valid',
                              opts.lab_dir, opts.codebooks_dir,
                              norm_dur=True,
                              max_spk_samples=opts.dur_max_samples,
                              parse_workers=opts.parser_workers,
                              max_seq_len=opts.dur_max_seq_len,
                              batch_size=opts.batch_size,
                              q_classes=opts.dur_q_classes)
        val_dloader = DataLoader(val_dset, batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.loader_workers, 
                                 collate_fn=varlen_dur_collate)
        if opts.dur_q_classes is not None:
            dur_outputs = opts.dur_q_classes
        else:
            dur_outputs = 1
        # build a duration model ready to train
        dur_model = sinout_duration(num_inputs=dset.ling_feats_dim,
                                    num_outputs=dur_outputs,
                                    emb_size=opts.dur_emb_size,
                                    rnn_size=opts.dur_rnn_size,
                                    rnn_layers=opts.dur_rnn_layers,
                                    sigmoid_out=opts.sigmoid_dur,
                                    dropout=opts.dur_dout,
                                    speakers=list(dset.spk2idx.keys()))
        adam = optim.Adam(dur_model.parameters(), lr=opts.dur_lr)
        if opts.cuda:
            dur_model.cuda()
        print('*' * 30)
        print('Built duration model')
        print(dur_model)
        print('*' * 30)
        patience = opts.patience
        if opts.dur_q_classes is not None:
            print('Num of dur classes: {}, using '
                  'NLLoss'.format(opts.dur_q_classes))
            criterion = F.nll_loss
        else:
            if opts.dur_loss == 'mse':
                criterion = F.mse_loss
            elif opts.dur_loss == 'mase':
                criterion = F.l1_loss
            elif opts.dur_loss == 'bce':
                if not opts.sigmoid_dur:
                    raise ValueError('Output has to be sigmoid if BCE is used')
                criterion = F.binary_cross_entropy
            else:
                raise ValueError('Dur loss not recognized: '
                                 '{}'.format(opts.dur_loss))
        tr_opts = {}
        if opts.dur_max_seq_len is not None:
            # we have a stateful approach
            tr_opts = {'stateful':True}
        train_engine(dur_model, dloader, adam, opts.log_freq, train_dur_epoch,
                     criterion,
                     opts.epoch, opts.save_path, 'dur_model.ckpt',
                     tr_opts=tr_opts,
                     eval_fn=eval_dur_epoch, val_dloader=val_dloader,
                     eval_stats=spk2durstats,
                     eval_target='total_nosil_dur_rmse',
                     eval_patience=opts.patience,
                     cuda=opts.cuda,
                     va_opts={'loss':'bce'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_spk', type=str, default='cfg/tcstar.cfg')
    parser.add_argument('--lab_dir', type=str, default='data/tcstar/lab')
    parser.add_argument('--codebooks_dir', type=str,
                        default='data/tcstar/codebooks.pkl')
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--force-gen', action='store_true',
                        default=False)
    parser.add_argument('--train-dur', action='store_true',
                        default=False,
                        help='Flag to specify that we train '
                             'a dur model.')
    parser.add_argument('--dur_max_samples', type=int, default=None,
                        help='Max samples per speaker in dur loader')
    parser.add_argument('--dur_rnn_size', type=int, default=256)
    parser.add_argument('--dur_rnn_layers', type=int, default=2)
    parser.add_argument('--dur_emb_size', type=int, default=256)
    parser.add_argument('--dur_q_classes', type=int, default=None,
                        help='Num of clusters in dur quantization. '
                             'If specified, this will triger '
                             'quantization in dloader and softmax '
                             'output for the model (Def: None).')
    parser.add_argument('--dur_loss', type=str, default='mse',
                        help='Options: mse | mae | bce')
    parser.add_argument('--sigmoid_dur', action='store_true', default=False)
    parser.add_argument('--dur_dout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dur_lr', type=float, default=0.001)
    parser.add_argument('--dur_max_seq_len', type=int, default=None)
    parser.add_argument('--loader_workers', type=int, default=2)
    parser.add_argument('--parser_workers', type=int, default=4)
    parser.add_argument('--cuda', default=False, action='store_true')

    opts = parser.parse_args()
    print('Parsed opts: ', json.dumps(vars(opts), indent=2))
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    main(opts)
