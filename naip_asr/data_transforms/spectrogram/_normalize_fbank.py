class NormalizeFbank(object):
    '''Normalize spectrogram using dataset mean and std'''
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
    
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        sample['fbank'] = fbank
        return sample