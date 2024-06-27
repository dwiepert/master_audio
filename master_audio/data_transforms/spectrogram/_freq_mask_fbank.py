import torch
import torchaudio
import matplotlib.pyplot as plt


class FreqMaskFbank(object):
    '''
    Frequency masking
    '''
    def __init__(self, freqm):
        self.freqm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freqm)
    
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = torch.transpose(fbank, 0, 1)
        #plt.pcolormesh(fbank)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)

        fbank = self.freqm(fbank)
        
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        sample['fbank'] = fbank

        return sample