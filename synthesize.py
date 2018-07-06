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
    if not opts.force_dur and opts.dur_model is None:
        raise ValueError('Please specify dur_model')

    if opts.aco_model is None:
        raise ValueError('Please specify aco_model')
    
    assert opts.model_cfg is not None
    with open(opts.model_cfg, 'r') as mcfg_f:
        mcfg = json.load(mcfg_f)
    device = 'cpu'
    if opts.cuda and torch.cuda.is_available():
        device = 'cuda'
    with open(opts.cfg_spk, 'rb') as cfg_f:
        cfg = pickle.load(cfg_f)
        idx2spk = {}
        spk2durstats = {}
        spk2acostats = {}
        for spk_id, spk_cfg in cfg.items():
            if 'idx' in spk_cfg:
                idx2spk[int(spk_cfg['idx'])] = spk_id
            if 'dur_stats' in spk_cfg:
                spk2durstats[int(spk_cfg['idx'])] = spk_cfg['dur_stats']
            if 'aco_stats' in spk_cfg:
                spk2acostats[int(spk_cfg['idx'])] = spk_cfg['aco_stats']

        with open(opts.codebooks_dir, 'rb') as cbooks_f:
            # read cbooks lengths to get ling_feats_dim
            cbooks = pickle.load(cbooks_f)
            ling_feats_dim = 0
            for k, v in cbooks.items():
                if 'mean' in v:
                    # real value
                    ling_feats_dim += 1
                else:
                    # categorical value
                    ling_feats_dim += len(v)
                # 6 boolean factors
            ling_feats_dim += 6
            print('Found ling_feats_dim: ', ling_feats_dim)
        if not opts.force_dur:
            print('-' * 30)
            print('Loading duration model: ', opts.dur_model)
            dur_model = torch.load(opts.dur_model, 
                                   map_location=lambda storage, loc: storage)
            print('[*] Loaded')
        else:
            print('[!] Dur model NOT loaded')
            dur_model = None
        print('>> spk2durstats: ', json.dumps(spk2durstats, indent=2))
        # build acoustic model and load weights
        print('-' * 30)
        print('Loading acoustic model: ', opts.aco_model)
        aco_model = torch.load(opts.aco_model,
                               map_location=lambda storage, loc: storage)
        print('[*] Loaded')
        #aco_model.load(opts.aco_model)
        print('>> idx2spk: ', json.dumps(idx2spk, indent=2))
        if opts.cuda:
            aco_model.to(device)
            if not opts.force_dur:
                dur_model.to(device)
        print('aco_model: ', aco_model)
        # get lab file basename
        lab_fname = os.path.basename(opts.synthesize_lab)
        lab_bname, _ = os.path.splitext(lab_fname)
        if mcfg['model_type'] in ['satt', 'decsatt']:
            att_synthesize(dur_model, aco_model, opts.spk_id, spk2durstats, spk2acostats,
                           opts.save_path, lab_bname, opts.codebooks_dir, opts.synthesize_lab, 
                           cuda=opts.cuda, force_dur=opts.force_dur, pf=opts.pf)
        else:
            synthesize(dur_model, aco_model, opts.spk_id, spk2durstats, spk2acostats,
                       opts.save_path, lab_bname, opts.codebooks_dir, opts.synthesize_lab, 
                       cuda=opts.cuda, force_dur=opts.force_dur, pf=opts.pf)


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
    parser.add_argument('--dur_emb_layers', type=int, default=1)
    parser.add_argument('--aco_rnn_size', type=int, default=512)
    parser.add_argument('--aco_rnn_layers', type=int, default=2)
    parser.add_argument('--aco_emb_size', type=int, default=256)
    parser.add_argument('--aco_emb_layers', type=int, default=2)
    parser.add_argument('--dur_q_classes', type=int, default=None,
                        help='Num of clusters in dur quantization. '
                             'If specified, this will triger '
                             'quantization in dloader and softmax '
                             'output for the model (Def: None).')
    #parser.add_argument('--dur_weights', type=str, default=None,
    #                    help='Trained dur model weights')
    parser.add_argument('--dur_model', type=str, default=None,
                        help='Trained dur model')
    parser.add_argument('--aco_model', type=str, default=None,
                        help='Trained aco model')
    #parser.add_argument('--aco_weights', type=str, default=None,
    #                    help='Trained aco model weights')
    parser.add_argument('--aco_q_classes', type=int, default=None,
                        help='Num of clusters in aco quantization. '
                             'If specified, this will triger '
                             'quantization in dloader and softmax '
                             'output for the model (Def: None).')
    parser.add_argument('--loss', type=str, default='MSELoss',
                        help='Options: PyTorch losses (Def: MSELoss)')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1991)
    parser.add_argument('--aco_bnorm', action='store_true', default=False)
    parser.add_argument('--aco_max_seq_len', type=int, default=None)
    parser.add_argument('--loader_workers', type=int, default=2)
    parser.add_argument('--parser_workers', type=int, default=4)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--dur_mulout', default=False, action='store_true')
    parser.add_argument('--aco_mulout', default=False, action='store_true')
    parser.add_argument('--exclude_train_spks', type=str, default=[], nargs='+')
    parser.add_argument('--exclude_eval_spks', type=str, default=[], nargs='+')
    parser.add_argument('--model_cfg', type=str, default=None)

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
