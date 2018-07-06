import torch.utils.data as data

class VCTK(data.Dataset):

    def __init__(self, vctk_root, maxlen=960, stride=0.5, 
                 cache_dir=None):
        super().__init__()
        self.vctk_root = vctk_root
        self.maxlen = maxlen
        self.stride = stride
        self.cache_dir = cache_dir


