import numpy as np

class ToNumpy(object):
    """
    Convert waveform to numpy
    """
    def __call__(self, sample):

        waveform = sample['waveform']
        sample['waveform'] = np.array(waveform)

        return sample