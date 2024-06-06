import numpy as np
import torch

class Mixup(object):
    '''
    Implement mixup of two files
    '''

    def __call__(self, sample, sample2=None):
        if sample2 is None:
            waveform = sample['waveform']
            waveform = waveform - waveform.mean()
        else:
            waveform1 = sample['waveform']
            waveform2 = sample2['waveform']
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0,0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0,0:waveform1.shape[1]]

            #sample lambda from beta distribution
            mix_lambda = np.random.beta(10,10)

            mix_waveform = mix_lambda*waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()   

            targets1 = sample['targets']
            targets2 = sample2['targets']
            targets = mix_lambda*targets1 + (1-mix_lambda)*targets2
            sample['targets'] = targets
            #TODO: what is happening here

        sample['waveform'] = waveform
