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
                          parse_workers=opts.parser_workers)
        spk2durstats = dset.spk2durstats
        print('Spk2durstats: ', json.dumps(spk2durstats, indent=2))
        dloader = DataLoader(dset, batch_size=opts.batch_size, shuffle=True,
                             num_workers=opts.loader_workers, 
                             collate_fn=varlen_dur_collate)
        # build validation dataset and loader
        val_dset = TCSTAR_dur(opts.cfg_spk, 'valid',
                              opts.lab_dir, opts.codebooks_dir,
                              force_gen=opts.force_gen,
                              norm_dur=True,
                              parse_workers=opts.parser_workers)
        val_dloader = DataLoader(val_dset, batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.loader_workers, 
                                 collate_fn=varlen_dur_collate)
        # build a duration model ready to train
        dur_model = sinout_duration(num_inputs=dset.ling_feats_dim,
                                    emb_size=opts.dur_emb_size,
                                    rnn_size=opts.dur_rnn_size,
                                    rnn_layers=opts.dur_rnn_layers,
                                    dropout=opts.dur_dout,
                                    speakers=list(dset.spk2idx.keys()))
        adam = optim.Adam(dur_model.parameters(), lr=0.001)
        if opts.cuda:
            dur_model.cuda()
        print('*' * 30)
        print('Built duration model')
        print(dur_model)
        print('*' * 30)
        tr_loss = []
        va_loss = []
        min_va_loss = 10e10
        patience = opts.patience
        for epoch in range(opts.epoch):
            best_model = False
            tr_loss += train_dur_epoch(dur_model, dloader, adam, opts.log_freq, epoch,
                                       cuda=opts.cuda)
            val_rmse = eval_dur_epoch(dur_model, val_dloader, 
                                      epoch, cuda=opts.cuda,
                                      spk2durstats=spk2durstats)
            if val_rmse < min_va_loss:
                print('Val loss improved {:.3f} -> {:.3f}'
                      ''.format(min_va_loss, val_rmse))
                min_va_loss = val_rmse
                best_model = True
                patience = opts.patience
            else:
                patience -= 1
                print('Val loss did not improve. Curr '
                      'patience: {}/{}'.format(patience,
                                               opts.patience))
                if patience == 0:
                    print('Out of patience. Ending DUR training.')
                    break
            dur_model.save(opts.save_path, 'dur_model.ckpt', epoch,
                          best_val=best_model)

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
    parser.add_argument('--dur_rnn_size', type=int, default=256)
    parser.add_argument('--dur_rnn_layers', type=int, default=2)
    parser.add_argument('--dur_emb_size', type=int, default=256)
    parser.add_argument('--dur_dout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--loader_workers', type=int, default=2)
    parser.add_argument('--parser_workers', type=int, default=4)
    parser.add_argument('--cuda', default=False, action='store_true')

    opts = parser.parse_args()
    print('Parsed opts: ', json.dumps(vars(opts), indent=2))
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    main(opts)
