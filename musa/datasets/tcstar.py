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
from sklearn.cluster import KMeans


def read_speaker_labs(spk_name, ids_list, lab_dir, lab_parser,
                      filter_by_dur=False):
    parsed_lines = [] # maintain seq structure
    parsed_tstamps = [] # maintain seq structure
    parse_timings = [] 
    flat_tstamps = []
    flat_lines = []
    beg_t = timeit.default_timer()
    if filter_by_dur:
        log_file = open('/tmp/dur_filter.log', 'w')
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
                    log_file.write('Filtred dur {} at file '
                                   '{}.lab\n'.format(dur, 
                                                 os.path.join(lab_dir,
                                                              spk_name,
                                                              split_id)))

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
        #parse_timings.append(timeit.default_timer() - beg_t)
        #print('Parsed spk-{} lab file {:5d}/{:5d}, mean time: {:.4f}'
        #      's'.format(spk_name, id_i, len(ids_list),
        #                 np.mean(parse_timings)),
        #     end='\n')
        #beg_t = timeit.default_timer()
    log_file.close()
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
    #print('batch[0][0][2] type: ', type(batch[0][0][2]))
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



class TCSTAR(Dataset):

    def __init__(self, spk_cfg_file, split, lab_dir,
                 lab_codebooks_path, force_gen=False,
                 ogmios_lab=True, parse_workers=4,
                 max_seq_len=None, batch_size=None,
                 max_spk_samples=None,
                 mulout=False):
        """
        # Arguments:
            max_seq_len: if specified, batches are stateful-like
                         with max_seq_len time-steps per sample,
                         and batch_size is also required.

            mulout: determines that speaker's data has to be
                    arranged in batches
        """
        if max_seq_len is not None:
            if batch_size is None:
                raise ValueError('Please specify a batch size in '
                                 ' TCSTAR to arrange the stateful '
                                 ' sequences.')
        self.max_seq_len = max_seq_len
        self.mulout = mulout
        self.batch_size = batch_size
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
        self.idx2spk = dict((v, k) for k, v in self.spk2idx.items())
        self.split = split
        self.lab_dir = lab_dir
        self.ogmios_lab = ogmios_lab
        self.force_gen = force_gen
        self.parse_workers = parse_workers
        self.lab_codebooks_path = lab_codebooks_path
        self.max_spk_samples = max_spk_samples
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
                 ogmios_lab=True, parse_workers=4, 
                 max_seq_len=None, batch_size=None,
                 max_spk_samples=None, 
                 mulout=False,
                 q_classes=None,
                 norm_dur=True):
        """
        # Arguments
            q_classes: integer specifying num of quantization clusters.
                       This means output will be given as integer, and the
                       clustering process is embedded in the data reading.
        """
        if q_classes is not None:
            assert isinstance(q_classes, int), type(q_classes)
        self.q_classes = q_classes
        self.norm_dur = norm_dur
        super(TCSTAR_dur, self).__init__(spk_cfg_file, split, lab_dir,
                                         lab_codebooks_path, force_gen=force_gen,
                                         ogmios_lab=ogmios_lab,
                                         parse_workers=parse_workers,
                                         max_seq_len=max_seq_len,
                                         mulout=mulout,
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
                          lab_parser, True)
            spk['result'] = parse_pool.apply_async(async_f, async_args)
        parse_pool.close()
        parse_pool.join()
        for sname, spk in self.speakers.items():
            result = spk['result'].get()
            parsed_timestamps = result[1]
            parsed_durs = tstamps_to_dur(parsed_timestamps)
            if self.norm_dur:
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
            if self.q_classes is not None:
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
            total_parsed_spks += [sname] * len(parsed_labs)
            del spk['result']
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
                    code = lab_enc(lab, normalize='minmax', sort_types=False)
                    # store reference to phoneme labels (to filter if needed)
                    phone_seq[t_] = lab[:5]
                    ndur = self.process_dur(spk, dur)
                    if spk not in all_durs:
                        all_durs[spk] = []
                    all_durs[spk].append(dur)
                    vec_seq[t_] = [self.spk2idx[spk], code, ndur]
                    if not hasattr(self, 'ling_feats_dim'):
                        self.ling_feats_dim = len(code)
                        print('setting ling feats dim: ', len(code))
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
            pickle.dump(all_durs, open('/tmp/durs.pickle', 'wb'))
        else:
            print('-' * 50)
            print('Encoding dur samples with max_seq_len {} and batch_size '
                  '{}'.format(self.max_seq_len, self.batch_size))
            # First, arrange all sequences into one very long one
            # Then split it into batch_size sequences, and arrange
            # samples to follow batch_size interleaved samples (stateful)
            all_code_seq = []
            all_phone_seq = []
            for spk, dur_seq, lab_seq in zip(total_parsed_spks,
                                             total_parsed_durs,
                                             total_parsed_labs):
                for t_, (dur, lab) in enumerate(zip(dur_seq, lab_seq)):
                    code = lab_enc(lab, normalize='minmax', sort_types=False)
                    ndur = self.process_dur(spk, dur)
                    all_code_seq.append([self.spk2idx[spk]] + code + [ndur])
                    all_phone_seq.append(lab[:5])
                    if not hasattr(self, 'ling_feats_dim'):
                        self.ling_feats_dim = len(code)
                        print('setting ling feats dim: ', len(code))
            # all_code_seq contains the large sequence of features (in, out)
            all_seq_len = len(all_code_seq)
            print('Lengt of all code_seq: ', all_seq_len)
            total_batches = (all_seq_len // (self.batch_size * \
                                             self.max_seq_len)) + 1
            print('total stateful batches: ', total_batches)
            # pad large sequences to operate with numpy arrays
            pad_len = total_batches * (self.batch_size * self.max_seq_len) - \
                      all_seq_len
            print('pad_len: ', pad_len)
            all_code_seq += [[0.] + [0.] * self.ling_feats_dim + [0]] * \
                            pad_len
            all_phone_seq += [['<PAD>'] * 5] * pad_len
            co_arr = np.array(all_code_seq)
            ph_arr = np.char.array(all_phone_seq)
            print('co_arr shape: ', co_arr.shape)
            print('ph_arr shape: ', ph_arr.shape)
            # interleave numpy samples
            co_arr = co_arr.reshape((self.batch_size, -1, co_arr.shape[-1]))
            ph_arr = ph_arr.reshape((self.batch_size, -1, ph_arr.shape[-1]))
            print('co_arr reshaped shape: ', co_arr.shape)
            print('ph_arr reshaped shape: ', ph_arr.shape)
            co_arr = np.split(co_arr, co_arr.shape[1] // self.max_seq_len, axis=1)
            ph_arr = np.split(ph_arr, ph_arr.shape[1] // self.max_seq_len, axis=1)
            print('Interleaved co_arr[0] shape: ', co_arr[0].shape)
            co_arr = np.concatenate(co_arr, axis=0)
            ph_arr = np.concatenate(ph_arr, axis=0)
            print('Interleaved co_arr: ', co_arr.shape)
            print('Interleaved ph_arr: ', ph_arr.shape)
            # make phone array a list again
            self.phone_sample = ph_arr.tolist() 
            # format data excluding padding symbols now
            for phone_sample, vec_sample in zip(ph_arr, co_arr):
                vec_seq = []
                ph_seq = []
                if phone_sample[0][0] == '<PAD>':
                    # This sample is just a remaining of padding
                    # do not include in samples
                    break
                for t_ in range(vec_sample.shape[0]):
                    vec_seq_el = vec_sample[t_]
                    ph_seq_el = phone_sample[t_]
                    if ph_seq_el[0] == '<PAD>':
                        print('Breaking list format loop at t_ {}'.format(t_))
                        break
                    dur_i_t = vec_seq_el[-1]
                    if self.q_classes is not None:
                        dur_i_t = np.array(dur_i_t, dtype=np.int64)
                    vec_seq.append([vec_seq_el[0], vec_seq_el[1:-1], 
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

    def __getitem__(self, index, spk_key=None):
        if isinstance(self.vec_sample, dict):
            # select hierarchicaly, first speaker, and then that speaker's sample
            if spk_key is None:
                raise IndexError('Accessing MO Dataset with SO format. Use the '
                                 'proper Sampler in your loader please.')
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
                 aco_window_stride=80, aco_window_len=320, 
                 aco_frame_rate=16000):
        super(TCSTAR_aco, self).__init__(spk_cfg_file, split, lab_dir,
                                         lab_codebooks_path, force_gen,
                                         ogmios_lab=ogmios_lab,
                                         parse_workers=parse_workers,
                                         max_seq_len=max_seq_len,
                                         batch_size=batch_size)
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
                # TODO: beware, not filtering durs
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



