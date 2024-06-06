import torch
import numpy as np

class FbankNoise(object):
    '''
    Add random noise to spectrogram
    '''
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        sample['fbank'] = fbank
        return sample