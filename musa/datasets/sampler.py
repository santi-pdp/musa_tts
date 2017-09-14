from torch.utils.data.sampler import Sampler
from random import shuffle
import json
import numpy as np


class MOSampler(Sampler):

    def __init__(self, spk2size, mo_dataset, 
                 batch_size,
                 randomize_rounds=False):
        """ 
        # Arguments
            spk2size: dict containing num of samples
                      per speaker id.
            mo_dataset: Dataset w/ MO instantiation.
            randomize_rounds: randomize id order per round or not.
        """
        self.spk2size = spk2size
        self.mo_dataset = mo_dataset
        self.batch_size = batch_size
        self.randomize_rounds = randomize_rounds
        print('Setting up MO sampler with spk sizes: ')
        print(json.dumps(self.spk2size, indent=2))

    def __iter__(self):
        # arrange indeces of a whole epoch, defining here
        # the rounds by interleaving speakers
        len_by_spk = self.mo_dataset.len_by_spk()
        assert isinstance(len_by_spk, dict)
        # ASSUMING ALL SPEAKERS HAVE SAME AMOUNT OF DATA
        # compute number of rounds
        spks = list(len_by_spk.keys())
        N = len_by_spk[spks[0]]
        #print('total N: ', N)
        # make a list of indices per speaker
        spk_idces = dict((k, list(range(N))) for k in spks)
        # shuffle every spk id
        for k, v in spk_idces.items():
            shuffle(spk_idces[k])
        # including smaller batch in the end
        n_rounds = int(np.ceil(N / self.batch_size))
        #print('num rounds: ', n_rounds)
        epoch_ids = []
        # make the list of indeces of every round
        for beg_i in range(0, N, self.batch_size):
            shuffle(spks)
            left = self.batch_size
            if N - beg_i < self.batch_size:
                left = N - beg_i
            for spkname in spks:
                #print('Fetching spk {} beg_i {}'.format(spkname, beg_i))
                batch = [(ii, spkname) for ii in \
                         range(beg_i,beg_i+left)]
                epoch_ids += batch
        return iter(epoch_ids)

    def __len__(self):
        return len(self.mo_dataset)


