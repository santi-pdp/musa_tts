import argparse
from musa.datasets import *
from musa.models import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from musa.core import *
import random
import json
import os


def main(opts):
    with open(os.path.join(opts.save_path, 
                           'main.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    # load criterion for both training options
    criterion = getattr(nn, opts.loss)(size_average=True)
    # build dataset and loader
    dset = TCSTAR_dur(opts.cfg_spk, 'train',
                      opts.lab_dir, opts.codebooks_dir,
                      force_gen=opts.force_gen,
                      norm_dur=True,
                      exclude_train_spks=opts.exclude_train_spks,
                      max_spk_samples=opts.max_samples,
                      parse_workers=opts.parser_workers,
                      max_seq_len=opts.max_seq_len,
                      batch_size=opts.batch_size,
                      q_classes=opts.q_classes,
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
                          q_classes=opts.q_classes,
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
    if opts.q_classes is not None:
        dur_outputs = opts.q_classes
    else:
        dur_outputs = 1
    if opts.dur_mulout:
        # select only available speakers to load, not all
        model_spks = list(dset.speakers.keys())
    else:
        model_spks = list(dset.all_speakers.keys())
    # build a duration model ready to train
    dur_model = sinout_duration(num_inputs=dset.ling_feats_dim,
                                num_outputs=dur_outputs,
                                emb_size=opts.emb_size,
                                rnn_size=opts.rnn_size,
                                rnn_layers=opts.rnn_layers,
                                sigmoid_out=opts.sigmoid_dur,
                                dropout=opts.dur_dout,
                                speakers=model_spks,
                                mulout=opts.dur_mulout,
                                cuda=opts.cuda,
                                emb_layers=opts.emb_layers)
    opti = getattr(optim, opts.dur_optim)(dur_model.parameters(),
                                      lr=opts.dur_lr)
    if opts.cuda:
        dur_model.cuda()
    print('*' * 30)
    print('Built duration model')
    print(dur_model)
    print('*' * 30)
    patience = opts.patience
    if opts.q_classes is not None:
        # Force NLLoss
        print('Num of dur classes: {}, using '
              'NLLoss'.format(opts.q_classes))
        criterion = nn.NLLoss()
    tr_opts = {'spk2durstats':spk2durstats,
               'idx2spk':dset.idx2spk}
    if opts.dur_max_seq_len is not None:
        # we have a stateful approach
        tr_opts['stateful'] = True
    va_opts = {'idx2spk':dset.idx2spk}
    if opts.dur_mulout:
        tr_opts['mulout'] = True
        va_opts['mulout'] = True
    if opts.q_classes is not None:
        va_opts['q_classes'] = True
    writer = SummaryWriter(os.path.join(opts.save_path,
                                        'train'))
    train_engine(dur_model, dloader, opti, opts.log_freq, train_dur_epoch,
                 criterion, opts.epoch, opts.save_path, 'dur_model.ckpt',
                 tr_opts=tr_opts,
                 eval_fn=eval_dur_epoch, val_dloader=val_dloader,
                 eval_stats=spk2durstats,
                 eval_target='total_nosil_dur_rmse',
                 eval_patience=opts.patience,
                 cuda=opts.cuda,
                 va_opts=va_opts,
                 log_writer=writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_spk', type=str, default='cfg/tcstar.cfg')
    parser.add_argument('--spk_id', type=int, default=0)
    parser.add_argument('--lab_dir', type=str, default='data/tcstar/lab')
    parser.add_argument('--aco_dir', type=str, default='data/tcstar/aco')
    parser.add_argument('--synthesize_lab', type=str, default=None,
                        help='Lab filename to be synthesized')
    parser.add_argument('--codebooks_dir', type=str,
                        default='data/tcstar/codebooks.pkl')
    parser.add_argument('--pf', type=float, default=1)
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--force-gen', action='store_true',
                        default=False)
    parser.add_argument('--force-dur', action='store_true',
                        default=False)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per speaker in dur loader')
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--emb_size', type=int, default=256)
    parser.add_argument('--emb_layers', type=int, default=1)
    parser.add_argument('--q_classes', type=int, default=None,
                        help='Num of clusters in dur quantization. '
                             'If specified, this will triger '
                             'quantization in dloader and softmax '
                             'output for the model (Def: None).')
    parser.add_argument('--dur_model', type=str, default=None,
                        help='Trained dur model')
    parser.add_argument('--loss', type=str, default='MSELoss',
                        help='Options: PyTorch losses (Def: MSELoss)')
    parser.add_argument('--sigmoid_dur', action='store_true', default=False)
    parser.add_argument('--dur_dout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1991)
    parser.add_argument('--dur_lr', type=float, default=0.001)
    parser.add_argument('--dur_optim', type=str, default='RMSprop')
    parser.add_argument('--emb_activation', type=str, default='Tanh')
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--loader_workers', type=int, default=2)
    parser.add_argument('--parser_workers', type=int, default=4)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--dur_mulout', default=False, action='store_true')
    parser.add_argument('--exclude_train_spks', type=str, default=[], nargs='+')
    parser.add_argument('--exclude_eval_spks', type=str, default=[], nargs='+')

    opts = parser.parse_args()
    print('Parsed opts: ', json.dumps(vars(opts), indent=2))
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)
    main(opts)
