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
                          exclude_train_spks=opts.exclude_train_spks,
                          max_spk_samples=opts.dur_max_samples,
                          parse_workers=opts.parser_workers,
                          max_seq_len=opts.dur_max_seq_len,
                          batch_size=opts.batch_size,
                          q_classes=opts.dur_q_classes,
                          mulout=opts.dur_mulout)
        # can be dur norm or kmeans data
        spk2durstats = dset.spk2durstats
        try:
            print('Spk2durstats: ', json.dumps(spk2durstats, indent=2))
        except TypeError as e:
            print(e)
        if opts.dur_mulout:
            sampler = MOSampler(dset.len_by_spk(), dset, opts.batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        dloader = DataLoader(dset, batch_size=opts.batch_size, shuffle=shuffle,
                             num_workers=opts.loader_workers, 
                             sampler=sampler,
                             collate_fn=varlen_dur_collate)
        # build validation dataset and loader
        val_dset = TCSTAR_dur(opts.cfg_spk, 'valid',
                              opts.lab_dir, opts.codebooks_dir,
                              norm_dur=True,
                              exclude_eval_spks=opts.exclude_eval_spks,
                              max_spk_samples=opts.dur_max_samples,
                              parse_workers=opts.parser_workers,
                              max_seq_len=opts.dur_max_seq_len,
                              batch_size=opts.batch_size,
                              q_classes=opts.dur_q_classes,
                              mulout=opts.dur_mulout)
        if opts.dur_mulout:
            va_sampler = MOSampler(val_dset.len_by_spk(), val_dset, opts.batch_size)
        else:
            va_sampler = None
        val_dloader = DataLoader(val_dset, batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.loader_workers, 
                                 sampler=va_sampler,
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
                                    speakers=list(dset.all_speakers.keys()),
                                    mulout=opts.dur_mulout,
                                    cuda=opts.cuda)
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
            elif opts.dur_loss == 'mae':
                criterion = F.l1_loss
            elif opts.dur_loss == 'bce':
                if not opts.sigmoid_dur:
                    raise ValueError('Output has to be sigmoid if BCE is used')
                criterion = F.binary_cross_entropy
            else:
                raise ValueError('Dur loss not recognized: '
                                 '{}'.format(opts.dur_loss))
        tr_opts = {'spk2durstats':spk2durstats,
                   'idx2spk':dset.idx2spk}
        if opts.dur_max_seq_len is not None:
            # we have a stateful approach
            tr_opts['stateful'] = True
        va_opts = {'idx2spk':dset.idx2spk}
        if opts.dur_mulout:
            tr_opts['mulout'] = True
            va_opts['mulout'] = True
        if opts.dur_q_classes is not None:
            va_opts['q_classes'] = True
        train_engine(dur_model, dloader, adam, opts.log_freq, train_dur_epoch,
                     criterion,
                     opts.epoch, opts.save_path, 'dur_model.ckpt',
                     tr_opts=tr_opts,
                     eval_fn=eval_dur_epoch, val_dloader=val_dloader,
                     eval_stats=spk2durstats,
                     eval_target='total_nosil_dur_rmse',
                     eval_patience=opts.patience,
                     cuda=opts.cuda,
                     va_opts=va_opts)
    if opts.train_aco:
        # build dataset and loader
        dset = TCSTAR_aco(opts.cfg_spk, 'train', opts.aco_dir,
                          opts.lab_dir, opts.codebooks_dir,
                          force_gen=opts.force_gen,
                          parse_workers=opts.parser_workers,
                          max_seq_len=opts.aco_max_seq_len,
                          batch_size=opts.batch_size,
                          max_spk_samples=opts.aco_max_samples,
                          mulout=opts.aco_mulout,
                          q_classes=opts.aco_q_classes,
                          trim_to_min=True,
                          norm_aco=True,
                          exclude_train_spks=opts.exclude_train_spks)
        # can be dur norm or kmeans data
        spk2acostats = dset.spk2acostats
        try:
            print('Spk2acostats: ', json.dumps(spk2acostats, indent=2))
        except TypeError as e:
            print(e)
        if opts.aco_mulout:
            sampler = MOSampler(dset.len_by_spk(), dset, opts.batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        dloader = DataLoader(dset, batch_size=opts.batch_size, shuffle=shuffle,
                             num_workers=opts.loader_workers, 
                             sampler=sampler,
                             collate_fn=varlen_aco_collate)
        # build validation dataset and loader
        val_dset = TCSTAR_aco(opts.cfg_spk, 'valid', opts.aco_dir,
                              opts.lab_dir, opts.codebooks_dir,
                              parse_workers=opts.parser_workers,
                              max_seq_len=opts.aco_max_seq_len,
                              batch_size=opts.batch_size,
                              max_spk_samples=opts.aco_max_samples,
                              mulout=opts.aco_mulout,
                              q_classes=opts.aco_q_classes,
                              trim_to_min=True,
                              norm_aco=True,
                              exclude_eval_spks=opts.exclude_eval_spks)
        if opts.aco_mulout:
            va_sampler = MOSampler(val_dset.len_by_spk(), val_dset, opts.batch_size)
        else:
            va_sampler = None
        val_dloader = DataLoader(val_dset, batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.loader_workers, 
                                 sampler=va_sampler,
                                 collate_fn=varlen_aco_collate)
        # TODO: hardcoded atm
        aco_outputs = 43
        # build an acoustic model ready to train
        aco_model = acoustic_rnn(num_inputs=dset.ling_feats_dim + 2,
                                 #num_outputs=aco_outputs,
                                 emb_size=opts.aco_emb_size,
                                 rnn_size=opts.aco_rnn_size,
                                 rnn_layers=opts.aco_rnn_layers,
                                 sigmoid_out=opts.sigmoid_aco,
                                 dropout=opts.aco_dout,
                                 speakers=list(dset.all_speakers.keys()),
                                 mulout=opts.aco_mulout,
                                 cuda=opts.cuda)
        adam = optim.Adam(aco_model.parameters(), lr=opts.aco_lr)
        if opts.cuda:
            aco_model.cuda()
        print('*' * 30)
        print('Built acoustic model')
        print(aco_model)
        print('*' * 30)
        patience = opts.patience
        if opts.aco_loss == 'mse':
            criterion = F.mse_loss
        elif opts.aco_loss == 'mae':
            criterion = F.l1_loss
        elif opts.aco_loss == 'bce':
            if not opts.sigmoid_dur:
                raise ValueError('Output has to be sigmoid if BCE is used')
            criterion = F.binary_cross_entropy
        else:
            raise ValueError('Aco loss not recognized: '
                             '{}'.format(opts.aco_loss))
        tr_opts = {'spk2acostats':spk2acostats,
                   'idx2spk':dset.idx2spk}
        va_opts = {'idx2spk':dset.idx2spk}
        if opts.aco_mulout:
            tr_opts['mulout'] = True
            va_opts['mulout'] = True
        # TODO: finish engine methods to train/eval acoustic model
        train_engine(aco_model, dloader, adam, opts.log_freq, train_aco_epoch,
                     criterion,
                     opts.epoch, opts.save_path, 'aco_model.ckpt',
                     tr_opts=tr_opts,
                     eval_fn=eval_aco_epoch, val_dloader=val_dloader,
                     eval_stats=spk2acostats,
                     #eval_target=['total_nosil_aco_mcd', 
                     #             'total_nosil_rmse',
                     #             'total_nosil_afpr'],
                     eval_target='total_nosil_aco_mcd',
                     eval_patience=opts.patience,
                     cuda=opts.cuda,
                     va_opts=va_opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_spk', type=str, default='cfg/tcstar.cfg')
    parser.add_argument('--lab_dir', type=str, default='data/tcstar/lab')
    parser.add_argument('--aco_dir', type=str, default='data/tcstar/aco')
    parser.add_argument('--codebooks_dir', type=str,
                        default='data/tcstar/codebooks.pkl')
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--force-gen', action='store_true',
                        default=False)
    parser.add_argument('--train-dur', action='store_true',
                        default=False,
                        help='Flag to specify that we train '
                             'a dur model.')
    parser.add_argument('--train-aco', action='store_true',
                        default=False,
                        help='Flag to specify that we train '
                             'an aco model.')
    parser.add_argument('--dur_max_samples', type=int, default=None,
                        help='Max samples per speaker in dur loader')
    parser.add_argument('--aco_max_samples', type=int, default=None,
                        help='Max samples per speaker in aco loader')
    parser.add_argument('--dur_rnn_size', type=int, default=256)
    parser.add_argument('--dur_rnn_layers', type=int, default=1)
    parser.add_argument('--dur_emb_size', type=int, default=256)
    parser.add_argument('--aco_rnn_size', type=int, default=256)
    parser.add_argument('--aco_rnn_layers', type=int, default=1)
    parser.add_argument('--aco_emb_size', type=int, default=256)
    parser.add_argument('--dur_q_classes', type=int, default=None,
                        help='Num of clusters in dur quantization. '
                             'If specified, this will triger '
                             'quantization in dloader and softmax '
                             'output for the model (Def: None).')
    parser.add_argument('--aco_q_classes', type=int, default=None,
                        help='Num of clusters in aco quantization. '
                             'If specified, this will triger '
                             'quantization in dloader and softmax '
                             'output for the model (Def: None).')
    parser.add_argument('--dur_loss', type=str, default='mse',
                        help='Options: mse | mae | bce')
    parser.add_argument('--aco_loss', type=str, default='mse',
                        help='Options: mse | mae | bce')
    parser.add_argument('--sigmoid_dur', action='store_true', default=False)
    parser.add_argument('--sigmoid_aco', action='store_true', default=False)
    parser.add_argument('--dur_dout', type=float, default=0.5)
    parser.add_argument('--aco_dout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1991)
    parser.add_argument('--dur_lr', type=float, default=0.001)
    parser.add_argument('--dur_max_seq_len', type=int, default=None)
    parser.add_argument('--aco_lr', type=float, default=0.001)
    parser.add_argument('--aco_max_seq_len', type=int, default=None)
    parser.add_argument('--loader_workers', type=int, default=2)
    parser.add_argument('--parser_workers', type=int, default=4)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--dur_mulout', default=False, action='store_true')
    parser.add_argument('--aco_mulout', default=False, action='store_true')
    parser.add_argument('--exclude_train_spks', type=str, default=[], nargs='+')
    parser.add_argument('--exclude_eval_spks', type=str, default=[], nargs='+')

    opts = parser.parse_args()
    print('Parsed opts: ', json.dumps(vars(opts), indent=2))
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)
    main(opts)
