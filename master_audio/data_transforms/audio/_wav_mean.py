class WaveMean(object):
    '''
    Subtract the mean from the waveform
    '''
    def __call__(self, sample):
        
        waveform = sample['waveform']
        waveform = waveform - waveform.mean()
        sample['waveform'] = waveform
        
        return sample