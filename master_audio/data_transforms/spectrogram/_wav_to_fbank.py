import torch
import torchaudio

class Wav2Fbank(object):
    '''
    Spectrogram conversion V2
    '''
    def __init__(self, target_length, melbins, tf_co, tf_shift, override_wave=False):
        self.target_length = target_length
        self.melbins = melbins
        self.tf_co = tf_co
        self.tf_shift = tf_shift
        self.override_wave = override_wave

    def __call__(self, sample):
        waveform = sample['waveform']
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sample['sample_rate'], use_energy=False,
            window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
            
        if self.tf_co is not None:
            fbank=torch.FloatTensor((self.tf_co(image=fbank.numpy()))['image'])
        
        if self.tf_shift is not None:
            fbank=torch.FloatTensor((self.tf_shift(image=fbank.numpy()))['image'])

        sample['fbank'] = fbank

        if self.override_wave:
            del sample['waveform']
        return sample