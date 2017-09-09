import torch
from torch.utils.data import Dataset
import json
import pickle
import os
import glob
import sys
from musa.ops import *
from .utils import *
import timeit
import numpy as np
import multiprocessing as mp


def read_speaker_labs(spk_name, ids_list, lab_dir, lab_parser):
    parsed_lines = [] # maintain seq structure
    parsed_tstamps = [] # maintain seq structure
    parse_timings = [] 
    flat_tstamps = []
    flat_lines = []
    beg_t = timeit.default_timer()
    for id_i, split_id in enumerate(ids_list, start=1):
        spk_lab_dir = os.path.join(lab_dir, spk_name)
        lab_f = os.path.join(spk_lab_dir, '{}.lab'.format(split_id))
        with open(lab_f) as lf:
            lab_lines = [l.rstrip() for l in lf.readlines()]
        tstamps, parsed_lab = lab_parser(lab_lines)
        parsed_tstamps.append(tstamps)
        parsed_lines.append(parsed_lab)
        flat_lines += parsed_lab
        flat_tstamps += tstamps
        #parse_timings.append(timeit.default_timer() - beg_t)
        #print('Parsed spk-{} lab file {:5d}/{:5d}, mean time: {:.4f}'
        #      's'.format(spk_name, id_i, len(ids_list),
        #                 np.mean(parse_timings)),
        #     end='\n')
        #beg_t = timeit.default_timer()
    return (spk_name, parsed_tstamps, parsed_lines, flat_lines)

def read_speaker_aco(spk_name, ids_list, aco_dir):
    aco_data = None 
    parse_timings = []
    beg_t = timeit.default_timer()
    for id_i, split_id in enumerate(ids_list, start=1):
        spk_aco_dir = os.path.join(aco_dir, spk_name)
        cc_f = os.path.join(spk_aco_dir, '{}.cc'.format(split_id))
        fv_f = os.path.join(spk_aco_dir, '{}.fv'.format(split_id))
        lf0_f = os.path.join(spk_aco_dir, '{}.lf0'.format(split_id))
        # load aco files
        cc = np.loadtxt(cc_f)
        fv = np.loadtxt(fv_f).reshape(-1, 1)
        lf0 = np.loadtxt(lf0_f)
        # make lf0 interpolation and obtain u/v flag
        i_lf0, uv = interpolation(lf0, 
        unvoiced_symbol=-10000000000.0)
        i_lf0 = i_lf0.reshape(-1, 1)
        uv = uv.reshape(-1, 1)
        # merge into aco structure
        if aco_data is None:
            aco_data = np.concatenate((cc, fv, i_lf0, uv), axis=1)
        else:
            aco_data_ = np.concatenate((cc, fv, i_lf0, uv), axis=1)
            aco_data = np.concatenate((aco_data, aco_data_), axis=0)
        parse_timings.append(timeit.default_timer() - beg_t)
        print('Parsed spk-{} aco file {:5d}/{:5d}, mean time: {:.4f}'
              's'.format(spk_name, id_i, len(ids_list),
                         np.mean(parse_timings)),
             end='\n')
        beg_t = timeit.default_timer()
    return (spk_name, aco_data)
    

def varlen_dur_collate(batch):
    """ Variable length dur collate function,
        compose the batch of sequences (lab, dur) 
        by padding to the longest one found
    """
    # traverse the batch looking for the longest seq 
    max_seq_len = 0
    for seq in batch:
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
    # build the batches of spk_idx, labs and durs
    # each sample in batch is a sequence!
    spks = np.zeros((len(batch), max_seq_len), dtype=np.int64)
    durs = np.zeros((len(batch), max_seq_len), dtype=np.float32)
    lab_len = len(batch[0][0][1])
    labs = np.zeros((len(batch), max_seq_len, lab_len), dtype=np.float32)
    # store each sequence length
    seqlens = np.zeros((len(batch),), dtype=np.int32)
    for ith, seq in enumerate(batch):
        spk_seq = []
        dur_seq = []
        lab_seq = []
        for tth, (spk_i, lab, dur) in enumerate(seq):
            spk_seq.append(spk_i)
            dur_seq.append(dur)
            lab_seq.append(lab)
        # padding left-side (past)
        spks[ith, :len(spk_seq)] = spk_seq
        labs[ith, :len(lab_seq)] = lab_seq
        durs[ith, :len(dur_seq)] = dur_seq
        seqlens[ith] = len(spk_seq)
    # compose tensors batching sequences
    spks = torch.from_numpy(spks)
    labs = torch.from_numpy(labs)
    durs = torch.from_numpy(durs)
    seqlens = torch.from_numpy(seqlens)
    return spks, labs, durs, seqlens



class TCSTAR(Dataset):

    def __init__(self, spk_cfg_file, split, lab_dir,
                 lab_codebooks_path, force_gen=False,
                 ogmios_lab=True, parse_workers=4):
        with open(spk_cfg_file, 'rb') as cfg_f:
            # load speakers config paths
            self.speakers = pickle.load(cfg_f)
        # store spk2idx
        fspk = list(self.speakers.keys())[0]
        if 'idx' not in self.speakers[fspk]:
            print('Indexing speakers with their ids...')
            # index speakers with integer ids
            self.spk2idx = dict((sname, i) for i, sname in
                                enumerate(self.speakers.keys()))
            for spk,idx in self.spk2idx.items():
                self.speakers[spk]['idx'] = idx
            print('Created ids: ', json.dumps(self.spk2idx, indent=2))
        else:
            # load existing indexation
            self.spk2idx = {}
            for spk in self.speakers.keys():
                self.spk2idx[spk] = self.speakers[spk]['idx']
            print('Loaded ids: ', json.dumps(self.spk2idx, indent=2))
        self.split = split
        self.lab_dir = lab_dir
        self.ogmios_lab = ogmios_lab
        self.force_gen = force_gen
        self.parse_workers = parse_workers
        self.lab_codebooks_path = lab_codebooks_path
        self.load_lab()
        # save stats in case anything changed
        with open(spk_cfg_file, 'wb') as cfg_f:
            # load speakers config paths
            pickle.dump(self.speakers, cfg_f)

    def load_lab(self):
        raise NotImplementedError



class TCSTAR_dur(TCSTAR):
    """ represent (lab, dur) tuples for building duration models """

    def __init__(self, spk_cfg_file, split, lab_dir,
                 lab_codebooks_path, force_gen=False,
                 ogmios_lab=True, parse_workers=4, norm_dur=True):
        self.norm_dur = norm_dur
        super(TCSTAR_dur, self).__init__(spk_cfg_file, split, lab_dir,
                                         lab_codebooks_path, force_gen=force_gen,
                                         ogmios_lab=ogmios_lab,
                                         parse_workers=parse_workers)

    def tstamps_to_dur(self, tstamps):
        # convert the list of lists of tstamps [[beg_1, end_1], ...] to durs
        # in seconds
        durs = []
        for seq in tstamps:
            durs_t = []
            for t_i, tss in enumerate(seq):
                beg_t, end_t = map(float, tss)
                durs_t.append((end_t - beg_t) / 1e7)
            durs.append(durs_t)
        return durs

    def load_lab(self):
        lab_codebooks_path = self.lab_codebooks_path
        # make the label parser
        if not os.path.exists(lab_codebooks_path) and \
           self.split != 'train':
            # codebooks are only made from train split, or we can create
            # them externally and pass them here (if we want any other
            # split/dataset codebooks)
            raise ValueError('Codebooks not found in {}: Either pass a '
                             'pre-built codebooks archive, or '
                             'load the train split to make the lab '
                             'codebooks first.'.format(lab_codebooks_path))
        # build parser to read labels
        lab_parser = label_parser(ogmios_fmt=self.ogmios_lab)
        self.lab_parser = lab_parser
        num_parsed = 0
        # store labs reference with speaker ID
        self.labs = []
        total_parsed_labs = []
        total_flat_labs = []
        total_parsed_durs = []
        total_parsed_spks = []
        # prepare a multi-processing pool to parse labels faster
        parse_pool = mp.Pool(self.parse_workers)
        beg_t = timeit.default_timer()
        num_labs_total = sum(len(spk[self.split]) for sname, spk in
                                 self.speakers.items())
        print('TCSTAR_dur-{} > Parsing {} labs from {} speakers. '
              'Num workers: {}...'.format(self.split, 
                                          num_labs_total,
                                          len(self.speakers),
                                          self.parse_workers))
        for sname, spk in self.speakers.items():
            async_f = read_speaker_labs
            async_args = (sname, spk[self.split], self.lab_dir,
                          lab_parser)
            spk['result'] = parse_pool.apply_async(async_f, async_args)
        parse_pool.close()
        parse_pool.join()
        for sname, spk in self.speakers.items():
            result = spk['result'].get()
            parsed_timestamps = result[1]
            parsed_durs = self.tstamps_to_dur(parsed_timestamps)
            if self.norm_dur:
                if self.split == 'train' and ('dur_stats' not in spk or \
                                              self.force_gen):
                    # if they do not exist (or force_gen) and it's train split
                    dur_min = np.min(parsed_durs)
                    dur_max = np.max(parsed_durs)
                    spk['dur_stats'] = {'min':dur_min,
                                        'max':dur_max}
                elif self.split != 'train' and 'dur_stats' not in spk:
                    raise ValueError('Dur stats not available in spk config, '
                                     'and norm_dur option was specified. Load '
                                     'train split to solve this issue, or '
                                     'pre-compute the stats.')
            parsed_labs = result[2]
            total_flat_labs += result[3]
            total_parsed_durs += parsed_durs
            total_parsed_labs += parsed_labs
            total_parsed_spks += [sname] * len(parsed_labs)
            del spk['result']
        # Build label encoder (codebooks will be made if they don't exist)
        lab_enc = label_encoder(codebooks_path=lab_codebooks_path,
                                lab_data=total_flat_labs,
                                force_gen=self.force_gen)
        self.lab_enc = lab_enc
        end_t = timeit.default_timer()
        print('TCSTAR_dur-{} > Loaded lab codebooks in {:.4f} '
              's'.format(self.split, end_t - beg_t))
        # Encode all lab contents
        # store vectorized sequences of samples of triplets (spk, lab, dur)
        self.vec_sample = []
        print('TCSTAR_dur-{} > Vectorizing {} sequences..'
              '.'.format(self.split,
                         len(total_parsed_durs)))
        beg_t = timeit.default_timer()
        for spk, dur_seq, lab_seq in zip(total_parsed_spks, total_parsed_durs, 
                                         total_parsed_labs):
            vec_seq = [None] * len(dur_seq)
            for t_, (dur, lab) in enumerate(zip(dur_seq, lab_seq)):
                code = lab_enc(lab, normalize='minmax', sort_types=False)
                if self.norm_dur:
                    dur_stats = self.speakers[spk]['dur_stats']
                    #print('dur_stats: ', dur_stats)
                    ndur = (dur - dur_stats['min']) / (dur_stats['max'] - \
                                                       dur_stats['min'])
                    # store ref to this speaker dur stats to denorm outside
                    if not hasattr(self, 'spk2durstats'):
                        self.spk2durstats = {}
                    self.spk2durstats[self.spk2idx[spk]] = dur_stats
                else:
                    ndur = dur
                vec_seq[t_] = [self.spk2idx[spk], code, ndur]
                if not hasattr(self, 'ling_feats_dim'):
                    self.ling_feats_dim = len(code)
            self.vec_sample.append(vec_seq)
        end_t = timeit.default_timer()
        print('TCSTAR_dur-{} > Vectorized dur samples in {:.4f} '
              's'.format(self.split, end_t - beg_t))
        # All labs + durs are vectorized and stored at this point

    def __getitem__(self, index):
        # return triplet (spk_idx, code, ndur)
        return self.vec_sample[index]

    def __len__(self):
        return len(self.vec_sample)


class TCSTAR_aco(TCSTAR):

    def __init__(self, spk_cfg_file, split, aco_dir, lab_dir,
                 lab_codebooks_path, force_gen=False,
                 ogmios_lab=True, parse_workers=4, 
                 aco_window_stride=80, aco_window_len=320, 
                 aco_frame_rate=16000):
        super(TCSTAR_aco, self).__init__(spk_cfg_file, split, lab_dir,
                                         lab_codebooks_path, force_gen,
                                         ogmios_lab=ogmios_lab,
                                         parse_workers=parse_workers)
        self.aco_window_stride = aco_window_stride
        self.aco_window_len = aco_window_len
        self.aco_frame_rate = aco_frame_rate
        self.aco_dir = aco_dir
        # load lab features
        self.load_lab(ogmios_lab)
        # load acoustic features for speakers
        self.load_aco()

    def load_lab(self, lab_codebooks_path, ogmios_lab):
        # make the label parser
        if not os.path.exists(lab_codebooks_path) and \
           self.split != 'train':
            raise ValueError('Run the train split to make the lab '
                             'codebooks first.')
        lab_parser = label_parser(ogmios_fmt=ogmios_lab)
        num_parsed = 0
        # store labs reference with speaker ID
        self.labs = []
        total_parsed_labs = []
        parse_pool = mp.Pool(self.parse_workers)
        beg_t = timeit.default_timer()
        print('Loading {} speakers lab codebooks...'.format(len(self.speakers)))
        for spk in self.speakers:
            for id_ in spk[self.split]:
                self.labs.append((spk['name'], id_))
            if not os.path.exists(lab_codebooks_path) or self.force_gen:
                async_f = read_speaker_labs
                async_args = (spk['name'], spk[self.split], self.lab_dir,
                              lab_parser)
                spk['result'] = parse_pool.apply_async(async_f, async_args)
        parse_pool.close()
        parse_pool.join()
        for spk in self.speakers:
            if not os.path.exists(lab_codebooks_path) or self.force_gen:
                result = spk['result'].get()
                parsed_timestamps = result[1]
                parsed_labs = result[2]
                num_parsed += len(parsed_labs)
                total_parsed_labs += parsed_labs
                del spk['result']
        # Build label encoder (train split will make codebooks in case it
        # does not exist)
        lab_enc = label_encoder(codebooks_path=lab_codebooks_path,
                                lab_data=total_parsed_labs,
                                force_gen=self.force_gen)
        self.lab_enc = lab_enc
        end_t = timeit.default_timer()
        print('Loaded lab codebooks in {} s'.format(end_t - beg_t))

    def load_aco(self):
        parse_pool = mp.Pool(self.parse_workers)
        beg_t = timeit.default_timer()
        self.acos = []
        print('Loading {} speakers aco stats...'.format(len(self.speakers)))
        for spk in self.speakers:
            if 'aco_stats' not in spk or self.force_gen:
                async_f = read_speaker_aco
                async_args = (spk['name'], spk[self.split], self.aco_dir,)
                spk['result'] = parse_pool.apply_async(async_f, async_args)
        parse_pool.close()
        parse_pool.join()
        for spk in self.speakers:
            if 'aco_stats' not in spk or self.force_gen:
                result = spk['result'].get()
                aco_data = result[1]
                # compute aco stats and store them, throwing away heavy data
                spk['aco_stats'] = {'min':np.min(aco_data, axis=0),
                                    'max':np.max(aco_data, axis=0),
                                    'mean': np.mean(aco_data, axis=0),
                                    'std':np.std(aco_data, axis=0)}
                del spk['result']
        end_t = timeit.default_timer()
        print('Loaded acos in {} s'.format(end_t - beg_t))



