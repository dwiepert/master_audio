import librosa 
import torchaudio

import torch.nn.functional as nn

class Pad(object):
    '''
    Pad audio
    :param pad_size: length to pad to
    :param mode: padding type
    :param value: value to pad with
    :param librosa: boolean indicating whether to use librosa
    '''
    def __init__(self, pad_size, mode='constant', value=0, librosa: bool = False):
        
        self.pad_size = pad_size
        self.mode = mode
        self.value = value
        self.librosa = librosa
        
    def __call__(self, sample):    
        waveform, sample_rate = sample['waveform']

        wave_size = waveform.shape
        print('TODO')
        pad_dim = (1, 1)

        if not self.librosa:
            padded_wave = nn.pad(waveform, pad_dim, mode=self.mode, value=self.value)
        else:
            print('CHECK AXIS')
            if self.mode == 'edge':
                padded_wave = librosa.util.fix_length(waveform, size=self.pad_size, axis=-1, mode=self.mode)
            else:
                padded_wave = librosa.util.fix_length(waveform, size=self.pad_size, axis=-1)

        sample['waveform'] = padded_wave
        
        return sample