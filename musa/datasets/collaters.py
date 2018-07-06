import numpy as np
import torch


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
    if len(batch[0]) > 2:
        lab_batch = [b[2] for b in batch]
    else:
        lab_batch = None
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

class Aco2Id_Collater(object):

    def __init__(self, spk2idx, accent2idx, gender2idx):
        self.spk2idx = spk2idx
        self.accent2idx = accent2idx
        self.gender2idx = gender2idx

    def __call__(self, batch):
        return self.collate_wav2wav(batch)

    def collate_wav2wav(self, batch):
        # samples are tuples (wav, txt, spkid)
        wav_0 = batch[0][0]
        if isinstance(wav_0, dict):
            # acoustic features rather than wav
            aco_maxlens = {}
            for k, v in wav_0.items():
                aco_maxlens[k] = 0
        else:
            wav_maxlen = 0
        for sample in batch:
            wav, txt, spkid, accent, gender = sample
            if isinstance(wav, dict):
                for k, v in wav.items():
                    if aco_maxlens[k] < v.size(0):
                        aco_maxlens[k] = v.size(0)
            else:
                if wav.size(0) > wav_maxlen:
                    wav_maxlen = wav.size(0)
         
        # now build the batch tensors
        if isinstance(wav_0, dict):
            aco_bs = {}
            for k, maxlen in aco_maxlens.items():
                aco_bs[k] = torch.FloatTensor(len(batch), maxlen, wav_0[k].size(-1))
        else:
            wav_b = torch.FloatTensor(len(batch), wav_maxlen)
        spkid_b = torch.LongTensor(len(batch), 1)
        accent_b = torch.LongTensor(len(batch), 1)
        gender_b = torch.LongTensor(len(batch), 1)
        # fill the batches
        for bidx, sample in enumerate(batch):
            wav, txt, spkid, accent, gender = sample
            if isinstance(wav, dict):
                for k, v in wav.items():
                    aco_bs[k][bidx] = pad(v, aco_maxlens[k])
            spkid_b[bidx] = self.spk2idx[spkid]
            accent_b[bidx] = self.accent2idx[accent]
            gender_b[bidx] = self.gender2idx[gender]
            
        if isinstance(wav_0, dict):
            return aco_bs, spkid_b, accent_b, gender_b
        else:
            return wav_b, spkid_b, accent_b, gender_b
