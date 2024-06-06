import torch 

class Truncate(object):
    '''
    Cut audio to specified length with optional offset
    :param length: length to trim to in terms of s
    :param offset: offset for clipping in terms of s
    '''
    def __init__(self, length, offset = 0):
        
        self.length = length
        self.offset = offset
        
    def __call__(self, sample):
        
        waveform = sample['waveform']
        sr = sample['sample_rate']
        frames = int(self.length*sr)

        waveform_offset = waveform[:, self.offset:]
        n_samples_remaining = waveform_offset.shape[1]
        
        if n_samples_remaining >= frames:
            waveform_trunc = waveform_offset[:, :frames]
        else:
            n_channels = waveform_offset.shape[0]
            n_pad = frames - n_samples_remaining
            channel_means = waveform_offset.mean(axis = 1).unsqueeze(1)
            waveform_trunc = torch.cat([waveform_offset, torch.ones([n_channels, n_pad])*channel_means], dim = 1)
            
        sample['waveform'] = waveform_trunc
        
        return sample