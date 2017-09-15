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
        # ASSUMING ALL SPEAKERS HAVE SAME AMOUNT OF DATA
        self.len_by_spks = self.mo_dataset.len_by_spk()
        assert isinstance(self.len_by_spks, dict)
        spks = list(self.len_by_spks.keys())
        self.N = self.len_by_spks[spks[0]]
        # compute number of rounds
        self.round_M = int(np.ceil(self.N / batch_size))
        print('Number of rounds: ', self.round_M)
        print('Setting up MO sampler with spk sizes: ')
        print(json.dumps(self.spk2size, indent=2))

    def __iter__(self):
        # arrange indeces of a whole epoch, defining here
        # the rounds by interleaving speakers
        spks = list(self.len_by_spks.keys())
        # make a list of indices per speaker
        spk_idces = dict((k, list(range(self.N))) for k in spks)
        # shuffle every spk id
        for k, v in spk_idces.items():
            shuffle(spk_idces[k])
        # including smaller batch in the end
        epoch_ids = []
        # make the list of indeces of every round
        for beg_i in range(0, self.N, self.batch_size):
            shuffle(spks)
            left = self.batch_size
            if self.N - beg_i < self.batch_size:
                left = self.N - beg_i
            for spkname in spks:
                #print('Fetching spk {} beg_i {}'.format(spkname, beg_i))
                batch = [(ii, spkname) for ii in \
                         range(beg_i,beg_i+left)]
                epoch_ids += batch
        return iter(epoch_ids)

    def __len__(self):
        return len(self.mo_dataset)


