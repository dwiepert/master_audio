import torch
import torchaudio

class TimeMaskFbank(object):
    '''
    Time masking
    '''
    def __init__(self, timem):
        self.timem = torchaudio.transforms.TimeMasking(timem)
    
    def __call__(self, sample):
        fbank = sample['fbank']
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)

        fbank = self.timem(fbank)
        
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        sample['fbank'] = fbank

        return sample