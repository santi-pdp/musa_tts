import torch
from torch.utils.data import Dataset
from .collaters import *
import json
import pickle
import os
import glob
import sys
from musa.ops import *
from .utils import *
import timeit
import struct
import numpy as np
import multiprocessing as mp
from sklearn.cluster import KMeans
import copy


def read_aco_file(spk_name, file_id, aco_dir):
    spk_aco_dir = os.path.join(aco_dir, spk_name)
    cc = read_bin_aco_file(os.path.join(spk_aco_dir, '{}.cc'.format(file_id)))
    fv = read_bin_aco_file(os.path.join(spk_aco_dir, '{}.fv'.format(file_id)))
    lf0 = read_bin_aco_file(os.path.join(spk_aco_dir, '{}.lf0'.format(file_id)))
    fv = fv.reshape((-1, 1))
    cc = cc.reshape((-1, 40))
    # make lf0 interpolation and obtain u/v flag
    i_lf0, uv = interpolation(lf0,
                              unvoiced_symbol=-10000000000.0)
    i_lf0 = i_lf0.reshape(-1, 1)
    uv = uv.reshape(-1, 1)
    #print('cc shape: ', cc.shape)
    # merge into aco structure
    aco_data = np.concatenate((cc, fv, i_lf0, uv), axis=1)
    return aco_data

def parse_lab_aco_correspondences(durs, aco_data):
    """ Find the matching of acoustic frames to 
        duration boundaries.
        An acoustic frame is within a phoneme if
        >= 50% of the sliding window is within
        the phoneme boundaries.
    """
    # sampling rate
    sr = 16000.
    curr_dur_idx = 0
    # set up curr boundary to be 0
    # convert dur into samples, knowing
    # sampling rate is 16kHz and dur is
    # in seconds
    #print('Parsing aco w/ durs: ', durs)
    #print('Parsing aco w/ acos shape: ', len(aco_data))
    cboundary = int(durs[curr_dur_idx] * sr)
    # keep track of curr ph dur to compute reldur
    curr_ph_dur = int(durs[curr_dur_idx] * sr)
    # keep track of acumulated boundaries for reldur
    acum_dur = 0
    # set up current centroid of window
    # in samples
    wind_t = 0
    wind_size = 320
    wind_stride = 80
    half_w = wind_size * .5
    aco_seq_data = [[]]
    # retrieve the tuples of relative durs (relative, absolute)
    reldurs = [[]]
    for aco_i in range(aco_data.shape[0]):
        if wind_t >= cboundary and curr_dur_idx < (len(durs) - 1):
            # window belongs to next phoneme, step on
            aco_seq_data.append([])
            reldurs.append([])
            curr_dur_idx += 1
            #print('wind_t > cboundary'
            #      ' ({}, {})'
            #      ''.format(wind_t, 
            #                cboundary))
            cboundary += int(durs[curr_dur_idx] * sr)
            acum_dur += curr_ph_dur
            curr_ph_dur = int(durs[curr_dur_idx] * sr)
            #print('Moving cboundary to {}'.format(cboundary))
            #print('durs len is: ', len(durs))
            #print('last cboundary will be: ', int(sum(durs) * sr))
        aco_seq_data[curr_dur_idx].append(aco_data[aco_i])
        # compute reldur within current ph dur
        reldur = (wind_t - acum_dur) / curr_ph_dur
        reldurs[curr_dur_idx].append([reldur, curr_ph_dur / sr])
        #print('Curr wind_t: {}, cboundary: {}, curr_dur_idx: '
        #      '{}, reldur: {}, curr_ph_dur: {},'
        #      'curr_ph_dur / sr: {}'.format(wind_t, 
        #                                    cboundary, 
        #                                    curr_dur_idx,
        #                                    reldur,
        #                                    curr_ph_dur,
        #                                    curr_ph_dur / sr))
        wind_t += wind_stride
    return aco_seq_data, reldurs

def read_speaker_labs(spk_name, ids_list, lab_dir, lab_parser,
                      filter_by_dur=False, aco_dir=None):
    parsed_lines = [] # maintain seq structure
    parsed_tstamps = [] # maintain seq structure
    if aco_dir is not None:
        parsed_aco = [] # aco data if parsed
        parsed_reldur = [] # reldur data
    parse_timings = [] 
    flat_tstamps = []
    flat_lines = []
    beg_t = timeit.default_timer()
    #if filter_by_dur:
        #log_file = open('/tmp/dur_filter.log', 'w')
    for id_i, split_id in enumerate(ids_list, start=1):
        spk_lab_dir = os.path.join(lab_dir, spk_name)
        lab_f = os.path.join(spk_lab_dir, '{}.lab'.format(split_id))
        with open(lab_f) as lf:
            lab_lines = [l.rstrip() for l in lf.readlines()]
        tstamps, parsed_lab = lab_parser(lab_lines)
        if filter_by_dur:
            filtered_lab = []
            filtered_tstamps = []
            # compute durs from timestamps to keep VALID phonemes
            converted_durs = tstamps_to_dur(tstamps, True)
            assert len(converted_durs) == len(parsed_lab), \
            len(converted_durs)
            for (plab, dur, tss) in zip(parsed_lab, converted_durs,
                                        tstamps):
                #print('dur=', dur)
                if dur > 0:
                    #print('ACCEPTED with dur: ', dur)
                    filtered_lab.append(plab)
                    filtered_tstamps.append(tss)
                #else:
                    #print('Filtered dur: ', dur)
                    #log_file.write('Filtred dur {} at file '
                    #               '{}.lab\n'.format(dur, 
                    #                             os.path.join(lab_dir,
                    #                                          spk_name,
                    #                                          split_id)))

            flat_lines += filtered_lab
            flat_tstamps += filtered_tstamps
            parsed_tstamps.append(filtered_tstamps)
            parsed_lines.append(filtered_lab)
            a_durs = len(filtered_tstamps) / len(converted_durs)
            #print('Ratio accepted durs: {}%'.format(a_durs * 100))
        else:
            parsed_tstamps.append(tstamps)
            parsed_lines.append(parsed_lab)
            flat_lines += parsed_lab
            flat_tstamps += tstamps
        if aco_dir is not None:
            #print('split_id: ', split_id)
            #print('parsed_tstamps: ', parsed_tstamps)
            #print('parsed_tstamps[-1]: ', parsed_tstamps[-1])
            # parse aco
            parsed_durs = tstamps_to_dur(parsed_tstamps[-1], True)
            aco_seq = read_aco_file(spk_name, split_id, aco_dir)
            #print('Total read aco frames: ', aco_seq.shape)
            aco_seq_data, \
            seq_reldur = parse_lab_aco_correspondences(parsed_durs, 
                                                       aco_seq)
            parsed_aco.append(aco_seq_data)
            parsed_reldur.append(seq_reldur)
        #parse_timings.append(timeit.default_timer() - beg_t)
        #print('Parsed spk-{} lab file {:5d}/{:5d}, mean time: {:.4f}'
        #      's'.format(spk_name, id_i, len(ids_list),
        #                 np.mean(parse_timings)),
        #     end='\n')
        #beg_t = timeit.default_timer()
    #log_file.close()
    if aco_dir is None:
        return (spk_name, parsed_tstamps, parsed_lines, flat_lines)
    else:
        return (spk_name, parsed_tstamps, parsed_lines, flat_lines, 
                parsed_aco, parsed_reldur)

def read_speaker_aco(spk_name, ids_list, aco_dir):
    aco_data = None 
    parse_timings = []
    beg_t = timeit.default_timer()
    for id_i, split_id in enumerate(ids_list, start=1):
        aco_data_ = read_aco_file(spk_name, split_id, aco_dir)
        # merge into aco structure
        if aco_data is None:
            aco_data = aco_data_
        else:
            aco_data = np.concatenate((aco_data, aco_data_), axis=0)
        parse_timings.append(timeit.default_timer() - beg_t)
        print('Parsed spk-{} aco file {:5d}/{:5d}, mean time: {:.4f}'
              's'.format(spk_name, id_i, len(ids_list),
                         np.mean(parse_timings)),
             end='\n')
        beg_t = timeit.default_timer()
    return (spk_name, aco_data)


class TCSTAR(Dataset):

    def __init__(self, spk_cfg_file, split, lab_dir,
                 lab_codebooks_path, force_gen=False,
                 ogmios_lab=True, parse_workers=4,
                 max_seq_len=None, batch_size=None,
                 max_spk_samples=None,
                 mulout=False, 
                 q_classes=None,
                 trim_to_min=False,
                 forced_trim=None,
                 exclude_train_spks=[],
                 exclude_eval_spks=[]):
        """
        # Arguments:
            spk_cfg_file: config file to read a dict
            split: 'train' 'valid' or 'test' split
            lab_dir: root lab dir with spks within
            lab_codebooks_path: codebooks file path dict
            force_gen: flag to enforce re-generation of codebooks
                       and stats.
            omgios_fmt: ogmios format to parse labs.
            max_seq_len: if specified, batches are stateful-like
                         with max_seq_len time-steps per sample,
                         and batch_size is also required.

            mulout: determines that speaker's data has to be
                    arranged in batches
            trim_to_min: trim all speakers to same num_samples if
                         maxlen is applied (specially for MO).
            forced_trim: max num of samples per speaker forced (this
                         has priority over trim_to_min counts)
        """
        self.trim_to_min = trim_to_min
        self.forced_trim = forced_trim
        if max_seq_len is not None:
            if batch_size is None:
                raise ValueError('Please specify a batch size in '
                                 ' TCSTAR to arrange the stateful '
                                 ' sequences.')
        else:
            print('WARNING: trim to min flag activated, but has no '
                  ' effect because no max_seq_len specified.')
        self.max_seq_len = max_seq_len
        if q_classes is not None:
            assert isinstance(q_classes, int), type(q_classes)
        self.q_classes = q_classes
        self.mulout = mulout
        self.batch_size = batch_size
        self.exclude_train_spks = exclude_train_spks
        with open(spk_cfg_file, 'rb') as cfg_f:
            # load speakers config paths
            self.speakers = pickle.load(cfg_f)
            self.all_speakers = copy.deepcopy(self.speakers)
            if split == 'train':
                # filter speakers in exclude list
                for spk in self.all_speakers.keys():
                    if spk in exclude_train_spks:
                        print('Excluding speaker {} from train '
                              'split'.format(spk))
                        del self.speakers[spk]
            if split == 'valid':
                # filter speakers in exclude list
                for spk in self.all_speakers.keys():
                    if spk in exclude_eval_spks:
                        print('Excluding speaker {} from valid '
                              'split'.format(spk))
                        del self.speakers[spk]
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
        self.idx2spk = dict((v, k) for k, v in self.spk2idx.items())
        self.split = split
        self.lab_dir = lab_dir
        self.ogmios_lab = ogmios_lab
        self.force_gen = force_gen
        self.parse_workers = parse_workers
        self.lab_codebooks_path = lab_codebooks_path
        self.max_spk_samples = max_spk_samples
        # call load_lab
        self.load_lab()
        # save stats in case anything changed
        with open(spk_cfg_file, 'wb') as cfg_f:
            # update original speakers, excluded ones in
            # train will be unmodified
            for spk, spkval in self.speakers.items():
                self.all_speakers[spk] = spkval
            # load speakers config paths
            pickle.dump(self.all_speakers, cfg_f)

    def load_lab(self):
        raise NotImplementedError


    def parse_labs(self, lab_parser, compute_dur_stats=False, 
                   compute_dur_classes=False, aco_dir=None):
        # if aco_dir is pecified, aco_data will be parsed
        # This is used by TCSTAR_aco
        total_parsed_labs = []
        total_flat_labs = []
        total_parsed_durs = []
        total_parsed_spks = []
        total_parsed_aco = []
        total_parsed_reldur = []
        # prepare a multi-processing pool to parse labels faster
        parse_pool = mp.Pool(self.parse_workers)
        num_labs_total = sum(len(spk[self.split]) for sname, spk in
                                 self.speakers.items())
        if self.max_spk_samples is not None:
            num_labs_total = self.max_spk_samples * len(self.speakers)
        print('TCSTAR_dur-{} > Parsing {} labs from {} speakers. '
              'Num workers: {}...'.format(self.split, 
                                          num_labs_total,
                                          len(self.speakers),
                                          self.parse_workers))
        for sname, spk in self.speakers.items():
            async_f = read_speaker_labs
            if self.max_spk_samples is not None:
                spk_samples = spk[self.split][:self.max_spk_samples]
            else:
                spk_samples = spk[self.split]
            async_args = (sname, spk_samples, self.lab_dir,
                          lab_parser, True, 
                          aco_dir)
            spk['result'] = parse_pool.apply_async(async_f, async_args)
        parse_pool.close()
        parse_pool.join()
        for sname, spk in self.speakers.items():
            result = spk['result'].get()
            parsed_timestamps = result[1]
            parsed_durs = tstamps_to_dur(parsed_timestamps)
            if compute_dur_stats:
            #if self.norm_dur:
                if self.split == 'train' and ('dur_stats' not in spk or \
                                              self.force_gen):
                    flat_durs = [fd for dseq in parsed_durs for fd in dseq]
                    # if they do not exist (or force_gen) and it's train split
                    dur_min = np.min(flat_durs)
                    assert dur_min > 0, dur_min
                    dur_max = np.max(flat_durs)
                    assert dur_max > 0, dur_max
                    assert dur_max > dur_min, dur_max
                    spk['dur_stats'] = {'min':dur_min,
                                        'max':dur_max}
                elif self.split != 'train' and 'dur_stats' not in spk:
                    raise ValueError('Dur stats not available in spk config, '
                                     'and norm_dur option was specified. Load '
                                     'train split to solve this issue, or '
                                     'pre-compute the stats.')
            if compute_dur_classes:
            #if self.q_classes is not None:
                if self.split == 'train' and ('dur_clusters' not in spk or \
                                              self.force_gen) and \
                   self.q_classes is not None:
                    flat_durs = [fd for dseq in parsed_durs for fd in dseq]
                    flat_durs = np.array(flat_durs)
                    flat_durs = flat_durs.reshape((-1, 1))
                    # make quantization for every user training data samples
                    dur_kmeans = KMeans(n_clusters=self.q_classes,
                                        random_state=0).fit(flat_durs)
                    self.dur_kmeans = dur_kmeans
                    # Normalization of dur is not necessary anymore with clusters
                    spk['dur_clusters'] = dur_kmeans
            parsed_labs = result[2]
            total_flat_labs += result[3]
            total_parsed_durs += parsed_durs
            total_parsed_labs += parsed_labs
            #print('len(parsed_labs) = ', len(parsed_labs))
            total_parsed_spks += [sname] * len(parsed_labs)
            if aco_dir is not None:
                total_parsed_aco += result[-2]
                total_parsed_reldur += result[-1]
                if self.split == 'train' and ('aco_stats' not in spk or \
                                              self.force_gen):
                    flat_acos = [fa for aseq in result[-2] for adur in aseq \
                                 for fa in adur]
                    #print('len(flat_acos)=', len(flat_acos))
                    #print('len(flat_acos[0])=', len(flat_acos[0]))
                    aco_min = np.min(flat_acos, axis=0)
                    aco_max = np.max(flat_acos, axis=0)
                    assert aco_min.shape[0] == len(flat_acos[0]), aco_min.shape
                    spk['aco_stats'] = {'min':aco_min,
                                        'max':aco_max}
                # dur stats are necessary for absolute duration normalization
                if self.split == 'train' and ('dur_stats' not in spk or \
                                              self.force_gen):
                    #print('len parsed_durs: ', len(result[-1]))
                    #flat_durs = [fd for dseq in parsed_durs for fd in dseq]
                    flat_durs = []
                    for dfile in result[-1]:
                        for dseq in dfile:
                            for dpho in dseq:
                                fd = dpho[1]
                                flat_durs.append(fd)
                    #print('len flat_durs: ', len(flat_durs))
                    #print('flat_durs: ', flat_durs)
                    # if they do not exist (or force_gen) and it's train split
                    dur_min = np.min(flat_durs)
                    #assert dur_min > 0, dur_min
                    dur_max = np.max(flat_durs)
                    #assert dur_max > 0, dur_max
                    assert dur_max > dur_min, dur_max
                    spk['dur_stats'] = {'min':dur_min,
                                        'max':dur_max}
            del spk['result']
        if aco_dir is None:
            return parsed_labs, total_flat_labs, total_parsed_durs, \
                   total_parsed_labs, total_parsed_spks
        else:
            return parsed_labs, total_flat_labs, total_parsed_durs, \
                   total_parsed_labs, total_parsed_spks, total_parsed_aco, \
                   total_parsed_reldur
