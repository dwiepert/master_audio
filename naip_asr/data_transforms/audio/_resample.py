import librosa 
import torchaudio

class ResampleAudio(object):
    '''
    Resample a waveform
    :param resample_rate: rate to resample to
    :param librosa: boolean indicating whether to use librosa
    '''
    def __init__(self, resample_rate: int = 16000, librosa: bool = False):
        
        self.resample_rate = resample_rate
        self.librosa = librosa
        
    def __call__(self, sample):    
        waveform, sample_rate = sample['waveform'], sample['sample_rate']
        if sample_rate != self.resample_rate:
            if self.librosa:
                transformed = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.resample_rate)
            else:
                transformed = torchaudio.transforms.Resample(sample_rate, self.resample_rate)(waveform)
            sample['waveform'] = transformed
            sample['sample_rate'] = self.resample_rate
        
        return sample