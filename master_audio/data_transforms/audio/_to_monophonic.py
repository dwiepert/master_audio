import torch

class ToMonophonic(object):
    '''
    Convert to monochannel with a reduce function (can alter based on how waveform is loaded)
    :param reduce_fn: function to use for reducing channels
    '''
    def __init__(self, reduce_fn):
        
        self.reduce_fn = reduce_fn
        
    def __call__(self, sample):
        
        waveform = sample['waveform']
        #print(waveform.shape)
        waveform_mono = self.reduce_fn(waveform)
        #print(waveform_mono)
        #print(waveform_mono.shape)
        
        if waveform_mono.shape != torch.Size([1, waveform.shape[1]]):
            raise ValueError(f'Result of reduce_fn wrong shape, expected [1, {waveform.shape[1]}], got [{waveform_mono.shape[0], waveform_mono.shape[1]}]')
            
        sample['waveform'] = waveform_mono
            
        return sample