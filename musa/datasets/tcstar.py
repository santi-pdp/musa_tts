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
                else:
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
    

def varlen_dur_collate(batch):
    """ Variable length dur collate function,
        compose the batch of sequences (lab, dur) 
        by padding to the longest one found
    """
    ph_batch = [b[1] for b in batch]
    batch = [b[0] for b in batch]
    # traverse the batch looking for the longest seq 
    max_seq_len = 0
    for seq in batch:
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
    # build the batches of spk_idx, labs and durs
    # each sample in batch is a sequence!
    spks = np.zeros((len(batch), max_seq_len), dtype=np.int64)
    #print('#batch[0][0][2] type: ', type(batch[0][0][2]))
    #print('np array dtype: ', batch[0][0][2].dtype)
    if batch[0][0][2].dtype == np.int64:
        #print('int64')
        durs = np.zeros((len(batch), max_seq_len), dtype=np.int64)
    else:
    #    print('float32')
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
    return spks, labs, durs, seqlens, ph_batch

def varlen_aco_collate(batch):
    """ Variable length aco collate function,
        compose the batch of sequences (lab+dur, aco) 
        by padding to the longest one found
    """
    ph_batch = [b[1] for b in batch]
    batch = [b[0] for b in batch]
    # traverse the batch looking for the longest seq 
    max_seq_len = 0
    for seq in batch:
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
    # build the batches of spk_idx, labs+durs and acos
    # each sample in batch is a sequence!
    spks = np.zeros((len(batch), max_seq_len), dtype=np.int64)
    #print('batch[0][0][2] type: ', type(batch[0][0][2]))
    #print('np array dtype: ', batch[0][0][2].dtype)
    aco_dim = len(batch[0][0][2])
    if batch[0][0][2].dtype == np.int64:
        #print('int64')
        acos = np.zeros((len(batch), max_seq_len, aco_dim), dtype=np.int64)
    else:
    #    print('float32')
        acos = np.zeros((len(batch), max_seq_len, aco_dim), dtype=np.float32)
    lab_len = len(batch[0][0][1])
    labs = np.zeros((len(batch), max_seq_len, lab_len), dtype=np.float32)
    # store each sequence length
    seqlens = np.zeros((len(batch),), dtype=np.int32)
    for ith, seq in enumerate(batch):
        spk_seq = []
        aco_seq = []
        lab_seq = []
        for tth, (spk_i, lab, aco) in enumerate(seq):
            spk_seq.append(spk_i)
            aco_seq.append(aco)
            lab_seq.append(lab)
        # padding left-side (past)
        spks[ith, :len(spk_seq)] = spk_seq
        labs[ith, :len(lab_seq)] = lab_seq
        acos[ith, :len(aco_seq)] = aco_seq
        seqlens[ith] = len(spk_seq)
    # compose tensors batching sequences
    spks = torch.from_numpy(spks)
    labs = torch.from_numpy(labs)
    acos = torch.from_numpy(acos)
    seqlens = torch.from_numpy(seqlens)
    return spks, labs, acos, seqlens, ph_batch

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
                                enumerate(self.all_speakers.keys()))
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


class TCSTAR_dur(TCSTAR):
    """ represent (lab, dur) tuples for building duration models """

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
                 exclude_eval_spks=[],
                 norm_dur=True):
        """
        # Arguments
            q_classes: integer specifying num of quantization clusters.
                       This means output will be given as integer, and the
                       clustering process is embedded in the data reading.
        """
        self.norm_dur = norm_dur
        super(TCSTAR_dur, self).__init__(spk_cfg_file, split, lab_dir,
                                         lab_codebooks_path, force_gen=force_gen,
                                         ogmios_lab=ogmios_lab,
                                         parse_workers=parse_workers,
                                         max_seq_len=max_seq_len,
                                         mulout=mulout,
                                         q_classes=q_classes,
                                         trim_to_min=trim_to_min,
                                         forced_trim=forced_trim,
                                         exclude_train_spks=exclude_train_spks,
                                         exclude_eval_spks=exclude_eval_spks,
                                         batch_size=batch_size,
                                         max_spk_samples=max_spk_samples)


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
        beg_t = timeit.default_timer()
        num_parsed = 0
        parsed_labs, total_flat_labs, \
        total_parsed_durs, total_parsed_labs, \
        total_parsed_spks = self.parse_labs(lab_parser, 
                                            compute_dur_stats=self.norm_dur,
                                            compute_dur_classes=(self.q_classes is \
                                                                 not None))
        # Build label encoder (codebooks will be made if they don't exist or
        # if they are forced)
        lab_enc = label_encoder(codebooks_path=lab_codebooks_path,
                                lab_data=total_flat_labs,
                                force_gen=self.force_gen)
        self.lab_enc = lab_enc
        end_t = timeit.default_timer()
        print('TCSTAR_dur-{} > Loaded lab codebooks in {:.4f} '
              's'.format(self.split, end_t - beg_t))
        # Encode all lab contents
        # store vectorized sequences of samples of triplets (spk, lab, dur)
        if self.mulout:
            # separated by spk idx
            self.vec_sample = {}
            self.phone_sample = {}
        else:
            # all merged together
            self.vec_sample = []
            # store reference to phone identities (str)
            self.phone_sample = []
        print('TCSTAR_dur-{} > Vectorizing {} sequences..'
              '.'.format(self.split,
                         len(total_parsed_durs)))
        all_durs = {} # tmp
        beg_t = timeit.default_timer()
        if self.max_seq_len is None:
            for spk, dur_seq, lab_seq in zip(total_parsed_spks, total_parsed_durs, 
                                             total_parsed_labs):
                vec_seq = [None] * len(dur_seq)
                phone_seq = [None] * len(dur_seq)
                for t_, (dur, lab) in enumerate(zip(dur_seq, lab_seq)):
                    code = lab_enc(lab, normalize='minmax', sort_types=False,
                                   verbose=False)
                    # store reference to phoneme labels (to filter if needed)
                    phone_seq[t_] = lab[:5]
                    ndur = self.process_dur(spk, dur)
                    if spk not in all_durs:
                        all_durs[spk] = []
                    all_durs[spk].append(dur)
                    vec_seq[t_] = [self.spk2idx[spk], code, ndur]
                    if not hasattr(self, 'ling_feats_dim'):
                        self.ling_feats_dim = len(code)
                        print('Setting ling feats dim: ', len(code))
                if self.mulout:
                    spk_name = spk
                    if spk_name not in self.vec_sample:
                        self.vec_sample[spk_name] = []
                        self.phone_sample[spk_name] = []
                    self.vec_sample[spk_name].append(vec_seq)
                    self.phone_sample[spk_name].append(phone_seq)
                else:
                    self.vec_sample.append(vec_seq)
                    self.phone_sample.append(phone_seq)
            #pickle.dump(all_durs, open('/tmp/durs.pickle', 'wb'))
        else:
            print('-' * 50)
            print('Encoding dur samples with max_seq_len {} and batch_size '
                  '{}'.format(self.max_seq_len, self.batch_size))
            # First, arrange all sequences into one very long one
            # Then split it into batch_size sequences, and arrange
            # samples to follow batch_size interleaved samples (stateful)
            all_code_seq = {}
            all_phone_seq = {}
            for spk, dur_seq, lab_seq in zip(total_parsed_spks,
                                             total_parsed_durs,
                                             total_parsed_labs):
                for t_, (dur, lab) in enumerate(zip(dur_seq, lab_seq)):
                    code = lab_enc(lab, normalize='minmax', sort_types=False)
                    ndur = self.process_dur(spk, dur)
                    if spk not in all_code_seq:
                        all_code_seq[spk] = []
                        all_phone_seq[spk] = []
                    all_code_seq[spk].append([self.spk2idx[spk]] +\
                                              code + [ndur])
                    all_phone_seq[spk].append(lab[:5])
                    if not hasattr(self, 'ling_feats_dim'):
                        self.ling_feats_dim = len(code)
                        print('setting ling feats dim: ', len(code))
            all_seq_len = {}
            if self.trim_to_min:
                # count each spks samples
                counts = {}
                counts_min = np.inf
                counts_spk = None
            for spkname, spkdata in all_code_seq.items():
                # all_code_seq contains the large sequence of features (in, out)
                all_seq_len[spkname] = len(spkdata)
                print('{}: Length of all code_seq: '
                      '{}'.format(spkname,
                                  all_seq_len[spkname]))
                # trim data to fit stateful arrangement
                total_batches = all_seq_len[spkname] // (self.batch_size * \
                                                         self.max_seq_len)
                tri_code_seq = spkdata[:self.batch_size * self.max_seq_len * \
                                        total_batches]
                all_code_seq[spkname] = tri_code_seq
                tri_phone_seq = all_phone_seq[spkname][:self.batch_size * \
                                                       self.max_seq_len * \
                                                       total_batches]
                all_phone_seq[spkname] = tri_phone_seq
                print('total stateful batches: ', total_batches)
                if total_batches <= 0:
                    raise ValueError('Not enough dur samples to statefulize '
                                     'with specified max_len ({}) and '
                                     'batch_size ({})'.format(self.max_seq_len,
                                                              self.batch_size))

                # Create dict of data to statefulize codes and phones data
                to_st_data = {'co':{'data':all_code_seq[spkname], 
                                    'np_class':np.array},
                              'ph':{'data':all_phone_seq[spkname],
                                    'np_class':np.char.array}}
                st_data = statefulize_data(to_st_data, self.batch_size,
                                           self.max_seq_len)
                co_arr = st_data['co']['st_data']
                ph_arr = st_data['ph']['st_data']

                # re-format data per speaker excluding padding symbols now
                for phone_sample, vec_sample in zip(ph_arr, co_arr):
                    vec_seq = []
                    ph_seq = []
                    for t_ in range(vec_sample.shape[0]):
                        vec_seq_el = vec_sample[t_]
                        ph_seq_el = phone_sample[t_]
                        dur_i_t = vec_seq_el[-1]
                        if self.q_classes is not None:
                            dur_i_t = np.array(dur_i_t, dtype=np.int64)
                        vec_seq.append([vec_seq_el[0], 
                                        vec_seq_el[1:-1], 
                                        dur_i_t])
                        ph_seq.append(ph_seq_el.tolist())
                    if self.mulout:
                        spk_name = self.idx2spk[vec_seq_el[0]] 
                        if spk_name not in self.vec_sample:
                            self.vec_sample[spk_name] = []
                            self.phone_sample[spk_name] = []
                        self.vec_sample[spk_name].append(vec_seq)
                        self.phone_sample[spk_name].append(ph_seq)
                    else:
                        self.vec_sample.append(vec_seq)
                        self.phone_sample.append(ph_seq)
                    if self.trim_to_min:
                        spk_name = self.idx2spk[vec_seq_el[0]]
                        if spk_name not in counts:
                            counts[spk_name] = 0
                        counts[spk_name] += 1
            if self.trim_to_min or self.forced_trim is not None:
                if self.forced_trim is not None:
                    counts_min = self.forced_trim + 1
                    counts_spk = 'Forced Trim'
                else:
                    for spk_name, cnt in counts.items():
                        if counts[spk_name] < counts_min:
                            counts_min = counts[spk_name]
                            counts_spk = spk_name
                print('-- Trimming speaker samples --')
                print('counts_min: ', counts_min)
                print('counts_spk: ', counts_spk)
                print('len self.vec_sample prior to '
                      'trim: ', len(self.vec_sample))
                self.vec_sample, \
                self.phone_sample = trim_spk_samples(self.vec_sample,
                                                     self.phone_sample,
                                                     counts_min,
                                                     self.mulout)
                print('len self.vec_sample after trim: ', len(self.vec_sample))
                if self.mulout:
                    # go over all speaker ids and trim to max amount of samples
                    for spkname, phsample in self.phone_sample.items():
                        print('-' * 60)
                        print('spk {} phsample len: '
                              '{}'.format(spkname, len(phsample)))
        print('-' * 50)
        end_t = timeit.default_timer()
        print('TCSTAR_dur-{} > Vectorized dur samples in {:.4f} '
              's'.format(self.split, end_t - beg_t))
        # All labs + durs are vectorized and stored at this point

    def process_dur(self, spk, dur):
        if not hasattr(self, 'spk2durstats'):
            self.spk2durstats = {}
        if self.norm_dur and not self.q_classes:
            dur_stats = self.speakers[spk]['dur_stats']
            #print('dur_stats: ', dur_stats)
            ndur = (dur - dur_stats['min']) / (dur_stats['max'] - \
                                               dur_stats['min'])
            if self.spk2idx[spk] not in self.spk2durstats:
                # store ref to this speaker dur stats to denorm outside
                self.spk2durstats[self.spk2idx[spk]] = dur_stats
        elif self.q_classes is not None:
            spk_clusters = self.speakers[spk]['dur_clusters']
            # TODO: can do batch prediction, but nvm atm
            ndur = spk_clusters.predict([[dur]])[0]
            ndur = np.array(ndur, dtype=np.int64)
            if self.spk2idx[spk] not in self.spk2durstats:
                self.spk2durstats[self.spk2idx[spk]] = spk_clusters
        else:
            ndur = dur
        return ndur

    def __getitem__(self, index):
        if isinstance(self.vec_sample, dict):
            # select hierarchicaly, first speaker, and then that speaker's sample
            if not isinstance(index, tuple):
                raise IndexError('Accessing MO Dataset with SO format. Use the '
                                 'proper Sampler in your loader please.')
            spk_key = index[1]
            index = index[0]
            return self.vec_sample[spk_key][index], \
                   self.phone_sample[spk_key][index]
        else:
            # return seq of triplets (spk_idx, code, ndur) and 
            # seq of (ph_id_str)
            return self.vec_sample[index], self.phone_sample[index]

    def __len__(self):
        if isinstance(self.vec_sample, dict):
            # sup up all keys length for final len on num of samples
            total_samples = sum(len(spk_samples) for spkname, spk_samples in
                                self.vec_sample.items())
        else:
            # directly compute list length
            total_samples = len(self.vec_sample)
        return total_samples

    def len_by_spk(self):
        if not isinstance(self.vec_sample, dict):
            raise TypeError('Cannot get len_by_spk w/ SO format')
        else:
            lens = dict((k, len(v)) for k, v in self.vec_sample.items())
            return lens


class TCSTAR_aco(TCSTAR):

    def __init__(self, spk_cfg_file, split, aco_dir, lab_dir,
                 lab_codebooks_path, force_gen=False,
                 ogmios_lab=True, parse_workers=4, 
                 max_seq_len=None, batch_size=None,
                 max_spk_samples=None, 
                 mulout=False, exclude_train_spks=[],
                 exclude_eval_spks=[],
                 q_classes=None,
                 trim_to_min=False,
                 forced_trim=None,
                 norm_aco=True,
                 aco_window_stride=80, aco_window_len=320, 
                 aco_frame_rate=16000):
        self.aco_window_stride = aco_window_stride
        self.aco_window_len = aco_window_len
        self.aco_frame_rate = aco_frame_rate
        self.aco_dir = aco_dir
        self.norm_aco = norm_aco
        super(TCSTAR_aco, self).__init__(spk_cfg_file, split, lab_dir,
                                         lab_codebooks_path, force_gen=force_gen,
                                         ogmios_lab=ogmios_lab,
                                         parse_workers=parse_workers,
                                         max_seq_len=max_seq_len,
                                         mulout=mulout,
                                         q_classes=q_classes,
                                         trim_to_min=trim_to_min,
                                         forced_trim=forced_trim,
                                         exclude_train_spks=exclude_train_spks,
                                         exclude_eval_spks=exclude_eval_spks,
                                         batch_size=batch_size,
                                         max_spk_samples=max_spk_samples)
        if self.max_seq_len is None:
            raise ValueError('TCSTAR_aco does not accept untrimmed seqs.'
                             'Please specify a max_seq_len')
        if self.q_classes is not None:
            raise NotImplementedError('KMeans output in TCSTAR_aco to be '
                                      'implemented.')

        # load lab features
        #self.load_lab(ogmios_lab)
        # load acoustic features for speakers
        #self.load_aco()

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
        beg_t = timeit.default_timer()
        num_parsed = 0
        parsed_labs, total_flat_labs, \
        total_parsed_durs, total_parsed_labs, \
        total_parsed_spks, \
        total_parsed_aco, \
        total_parsed_reldur = self.parse_labs(lab_parser, 
                                              compute_dur_stats=False,
                                              compute_dur_classes=False,
                                              aco_dir=self.aco_dir)

        # Build label encoder (codebooks will be made if they don't exist or
        # if they are forced)
        lab_enc = label_encoder(codebooks_path=lab_codebooks_path,
                                lab_data=total_flat_labs,
                                force_gen=self.force_gen)
        self.lab_enc = lab_enc
        end_t = timeit.default_timer()
        print('TCSTAR_aco-{} > Loaded lab codebooks in {:.4f} '
              's'.format(self.split, end_t - beg_t))
        # Encode all lab contents
        # store vectorized sequences of samples of triplets (spk, lab, dur)
        if self.mulout:
            # separated by spk idx
            self.vec_sample = {}
            self.phone_sample = {}
        else:
            # all merged together
            self.vec_sample = []
            # store reference to phone identities (str)
            self.phone_sample = []
        total_aco_seqs = 0
        for aco_ in total_parsed_aco:
            total_aco_seqs += len(aco_[0]) * len(aco_[1])
        print('TCSTAR_aco-{} > Vectorizing {} sequences..'
              '.'.format(self.split,
                         total_aco_seqs))
        print('-' * 50)
        print('Encoding aco samples with max_seq_len {} and batch_size '
              '{}'.format(self.max_seq_len, self.batch_size))
        # First, arrange all sequences into very long ones (per spk)
        # Then split it into batch_size sequences, and arrange
        # samples to follow batch_size interleaved samples (stateful)
        all_code_seq = {}
        all_phone_seq = {}
        for spk, dur_seq, lab_seq, aco_seq, \
            reldur_seq in zip(total_parsed_spks,
                              total_parsed_durs,
                              total_parsed_labs,
                              total_parsed_aco,
                              total_parsed_reldur):
            #print('len(spk)=', len(spk))
            #print('len(dur_seq)=', len(dur_seq))
            #print('len(lab_seq)=', len(lab_seq))
            #print('len(aco_seq)=', len(aco_seq))
            #print('len(reldur_seq)=', len(reldur_seq))
            for t_, (dur, lab, aco, reldur) in enumerate(zip(dur_seq, 
                                                             lab_seq, 
                                                             aco_seq,
                                                             reldur_seq)):
                #print('len(aco) = ', len(aco))
                #print('len(reldur) = ', len(reldur))
                #code = lab_enc(lab, normalize='minmax', sort_types=False)
                code = lab_enc(lab, normalize='znorm', sort_types=False)
                for aco_ph, reldur_ph in zip(aco, reldur):
                    #print('len(reldur_ph)=', len(reldur_ph))
                    #print('reldur_ph[0]=', reldur_ph[0])
                    #print('reldur_ph[1]=', reldur_ph[1])
                    # process aco outputs and absolute dur
                    naco, nreldur = self.process_aco(spk, aco_ph, reldur_ph[1])
                    #print('aco_ph = ', aco_ph)
                    #print('naco = ', naco)
                    #print('reldur_ph[1] = ', reldur_ph[1])
                    #print('nreldur = ', nreldur)
                    nreldur = [reldur_ph[0], nreldur]
                    if spk not in all_code_seq:
                        all_code_seq[spk] = []
                        all_phone_seq[spk] = []
                    all_code_seq[spk].append([self.spk2idx[spk]] + code + nreldur + \
                                              naco.tolist())
                    all_phone_seq[spk].append(lab[:5])
                    if not hasattr(self, 'ling_feats_dim'):
                        self.ling_feats_dim = len(code)
                        print('setting ACO ling feats dim: ',
                              self.ling_feats_dim)
                    if not hasattr(self, 'aco_feats_dim'):
                        self.aco_feats_dim = len(naco)
                        print('setting ACO aco feats dim: ', len(naco))
        all_seq_len = {}
        if self.trim_to_min:
            # count each spks samples
            counts = {}
            counts_min = np.inf
            counts_spk = None
            #samples_cnt, min_cnt, min_spk = count_spk_samples(co_arr, ph_arr)
            #print('samples_cnt: ', json.dumps(samples_cnt, indent=2))
            #print('min_cnt: ', min_cnt)
            #print('min_spk: ', min_spk)
        for spkname, spkdata in all_code_seq.items():
            # all_code_seq contains the large sequence of features (in, out)
            all_seq_len[spkname] = len(spkdata)
            print('{}: Length of all code_seq: '
                  '{}'.format(spkname,
                              all_seq_len[spkname]))
            # trim data to fit stateful arrangement
            total_batches = all_seq_len[spkname] // (self.batch_size * \
                                                     self.max_seq_len)
            tri_code_seq = spkdata[:self.batch_size * self.max_seq_len * \
                                    total_batches]
            all_code_seq[spkname] = tri_code_seq
            tri_phone_seq = all_phone_seq[spkname][:self.batch_size * \
                                                   self.max_seq_len * \
                                                   total_batches]
            all_phone_seq[spkname] = tri_phone_seq
            print('total stateful batches: ', total_batches)

            # Create dict of data to statefulize codes and phones data
            to_st_data = {'co':{'data':all_code_seq[spkname], 
                                'np_class':np.array},
                          'ph':{'data':all_phone_seq[spkname],
                                'np_class':np.char.array}}
            st_data = statefulize_data(to_st_data, self.batch_size,
                                       self.max_seq_len)
            co_arr = st_data['co']['st_data']
            ph_arr = st_data['ph']['st_data']

            # re-format data per speaker excluding padding symbols now
            for phone_sample, vec_sample in zip(ph_arr, co_arr):
                vec_seq = []
                ph_seq = []
                for t_ in range(vec_sample.shape[0]):
                    vec_seq_el = vec_sample[t_]
                    ph_seq_el = phone_sample[t_]
                    aco_i_t = vec_seq_el[-self.aco_feats_dim:]
                    #if self.q_classes is not None:
                    #    aco_i_t = np.array(aco_i_t, dtype=np.int64)
                    vec_seq.append([vec_seq_el[0], vec_seq_el[1:-self.aco_feats_dim], 
                                    aco_i_t])
                    ph_seq.append(ph_seq_el.tolist())
                if self.mulout:
                    spk_name = self.idx2spk[vec_seq_el[0]] 
                    if spk_name not in self.vec_sample:
                        self.vec_sample[spk_name] = []
                        self.phone_sample[spk_name] = []
                    self.vec_sample[spk_name].append(vec_seq)
                    self.phone_sample[spk_name].append(ph_seq)
                else:
                    self.vec_sample.append(vec_seq)
                    self.phone_sample.append(ph_seq)
                if self.trim_to_min:
                    spk_name = self.idx2spk[vec_seq_el[0]]
                    if spk_name not in counts:
                        counts[spk_name] = 0
                    counts[spk_name] += 1
        if self.trim_to_min or self.forced_trim is not None:
            if self.forced_trim is not None:
                counts_min = self.forced_trim + 1
                counts_spk = 'Forced Trim'
            else:
                for spk_name, cnt in counts.items():
                    if counts[spk_name] < counts_min:
                        counts_min = counts[spk_name]
                        counts_spk = spk_name
            print('-- Trimming speaker samples --')
            print('counts_min: ', counts_min)
            print('counts_spk: ', counts_spk)
            print('len self.vec_sample prior to trim: ', len(self.vec_sample))
            self.vec_sample, \
            self.phone_sample = trim_spk_samples(self.vec_sample,
                                                 self.phone_sample,
                                                 counts_min,
                                                 self.mulout)
            print('len self.vec_sample after trim: ', len(self.vec_sample))
            first_vec_len = len(self.vec_sample[0])
            raw_aco = []
            for vi, vseq in enumerate(self.vec_sample):
                for vsample in vseq:
                    raw_aco.append(vsample[1])
            #np.save('/tmp/{}-aco.npy'.format(self.split), raw_aco)
            if self.mulout:
                # go over all speaker ids and trim to max amount of samples
                for spkname, phsample in self.phone_sample.items():
                    print('-' * 60)
                    print('spk {} phsample len: {}'.format(spkname, len(phsample)))
                    #for phs in phsample:
                    #    print('{}|{}'.format(spkname, phs[2]), end=' ')
        print('-' * 50)
        end_t = timeit.default_timer()
        print('TCSTAR_aco-{} > Vectorized dur samples in {:.4f} '
              's'.format(self.split, end_t - beg_t))
        # All labs + durs are vectorized and stored at this point
        # store labs reference with speaker ID

    def process_aco(self, spk, aco, dur):
        if not hasattr(self, 'spk2acostats'):
            self.spk2acostats = {}
        if self.norm_aco and not self.q_classes:
            aco_stats = self.speakers[spk]['aco_stats']
            dur_stats = self.speakers[spk]['dur_stats']
            naco = (aco - aco_stats['min']) / (aco_stats['max'] - \
                                               aco_stats['min'])
            ndur = (dur - dur_stats['min']) / (dur_stats['max'] - \
                                                dur_stats['min'])
            if self.spk2idx[spk] not in self.spk2acostats:
                # store ref to this speaker aco+dur stats to denorm outside
                self.spk2acostats[self.spk2idx[spk]] = {'dur':dur_stats,
                                                        'aco':aco_stats}
        elif self.q_classes is not None:
            """
            spk_clusters = self.speakers[spk]['dur_clusters']
            # TODO: can do batch prediction, but nvm atm
            ndur = spk_clusters.predict([[dur]])[0]
            ndur = np.array(ndur, dtype=np.int64)
            if self.spk2idx[spk] not in self.spk2durstats:
                self.spk2durstats[self.spk2idx[spk]] = spk_clusters
            """
            raise NotImplementedError
        else:
            naco = aco
            ndur = dur
        return naco, ndur

    def __getitem__(self, index):
        try:
            if isinstance(self.vec_sample, dict):
                # select hierarchicaly, first speaker, and then that speaker's sample
                if not isinstance(index, tuple):
                    raise IndexError('Accessing MO Dataset with SO format. Use the '
                                     'proper Sampler in your loader please.')
                spk_key = index[1]
                index = index[0]
                return self.vec_sample[spk_key][index], \
                       self.phone_sample[spk_key][index]
            else:
                # return seq of triplets (spk_idx, code+dur, aco) and 
                # seq of (ph_id_str)
                return self.vec_sample[index], self.phone_sample[index]
        except IndexError:
            print('Error accessing {}:{}'.format(spk_key, index))
            raise

    def __len__(self):
        if isinstance(self.vec_sample, dict):
            # sup up all keys length for final len on num of samples
            total_samples = sum(len(spk_samples) for spkname, spk_samples in
                                self.vec_sample.items())
        else:
            # directly compute list length
            total_samples = len(self.vec_sample)
        return total_samples

    def len_by_spk(self):
        if not isinstance(self.vec_sample, dict):
            raise TypeError('Cannot get len_by_spk w/ SO format')
        else:
            lens = dict((k, len(v)) for k, v in self.vec_sample.items())
            return lens
