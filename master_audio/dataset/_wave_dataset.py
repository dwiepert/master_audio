#IMPORTS
import random
from typing import List, Union

#third party
from audiomentations import *
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional
from torch.utils.data import Dataset


#local
from master_audio.data_transforms.audio import *
from master_audio.data_transforms.common import *
from master_audio.data_transforms.spectrogram import *
from master_audio.constants import *
     
class WaveDataset(Dataset):
    def __init__(self, data:Union[pd.DataFrame, np.ndarray],  prefix:str, model_type:str, model_task:str,
                 dataset_config:dict,target_labels:str=None, bucket=None):
        '''
        Dataset that manages audio recordings. 

        :param data: either an np.array of uids or dataframe with uids as index and annotations in the columns. For classification, must give a dataframe
        :param prefix: location of files to download (compatible with gcs)
        :param model_type: type of model this Dataset will be used with (e.g. w2v2, whisper)
        :param model_task: model task for this Dataset (e.g. asr, classification)
        :param dataset_config:dictionary with transform parameters 
        :param target_labels: str list of targets to extract from data. Can be none only for 'asr'.
        :param bucket: gcs bucket
        '''

        self.data = data
        self.target_labels = target_labels
        self.dataset_config = dataset_config
        self.bucket = bucket
        
        self.prefix = prefix
        self.model_type = model_type
        self.model_task = model_task


        self.use_librosa = self._check_existence('use_librosa')
        if self.use_librosa is None:
            self.use_librosa = False

        self.mixup = self._check_existence('mixup')
         

        if self.model_task == 'asr' or self.model_task == 'similarity':
            self.transforms = self._get_asr_transforms()

        
        if self.model_task == 'classification':
            if self.model_type == 'ssast':
                self.to_spectrogram = True
            elif self.model_type=='w2v2':
                self.to_spectrogram = False 
            else:
                raise NotImplementedError()
        
            assert self.target_labels is not None, 'Target labels must be given for classification.'
            assert isinstance(self.data, pd.DataFrame), 'Must give a dataframe of uids and annotations for classification.'
            self.transforms = self._get_clf_transforms()

        self.feature_extractor = self._check_existence('feature_extractor')
        if self.model_task == 'classification' and self.model_type == 'w2v2':
            assert self.feature_extractor is not None, 'W2V2 Classification model must have a feature extractor included in the dataset configuration.'
       
    
    def _check_existence(self, key):
        """
        Check if item exists in a dataset config
        """
        if key in self.dataset_config:
            return self.dataset_config.get(key)
        else:
            return None
        
    def _get_asr_transforms(self):
        """
        Get transforms for ASR task
        """
        self.load_waveform = self._check_existence('load_waveform')
        if self.load_waveform is None:
            self.load_waveform = False
        self.to_numpy = True
        self.to_tensor = False
        self.data_augmentation = False
        self.to_spectrogram = False
        if self.load_waveform:
            self.resample_rate = self._check_existence('resample_rate')
            self.monochannel = self._check_existence('monochannel')
            self.clip_length = self._check_existence('clip_length')
            self.trim_level = self._check_existence('trim_level')
            
            self.al_transform = []
            return self._getaudiotransforms()

        else:
            self.savedir = self._check_existence('savedir')
            if self.bucket is not None:
                assert self.savedir is not None, 'Save dir is necessary for cloud'
            return [torchvision.transforms.Compose([UidToPath(prefix=self.prefix, savedir=self.savedir, bucket=self.bucket)]), []]
    

    def _get_clf_transforms(self):
        """
        Get transforms for Classification task
        """
        self.resample_rate = self._check_existence('resample_rate')
        self.monochannel = self._check_existence('monochannel')
        self.clip_length = self._check_existence('clip_length')
        self.trim_level = self._check_existence('trim_level')
        self.to_numpy=False
        self.to_tensor = True
        # self.to_spectrogram = self._check_existence('to_spectrogram')
        # if self.to_spectrogram is None:
        #     self.to_spectrogram = False

        self.gauss = self._check_existence('gauss')
        self.gausssnr = self._check_existence('gausssnr')
        self.alias = self._check_existence('alias')
        self.bandstop = self._check_existence('bandstop')
        self.bitcrush = self._check_existence('bitcrush')
        self.clipd = self._check_existence('clipd')
        self.gain  = self._check_existence('gain')
        self.gaint= self._check_existence('gaint')
        self.mp3= self._check_existence('mp3')
        self.norm= self._check_existence('norm')
        self.pshift= self._check_existence('pshift')
        self.pinversion= self._check_existence('pinversion')
        self.tstretch= self._check_existence('tstretch')
        self.tmask= self._check_existence('tmask')
        self.tanh = self._check_existence('tanh')
        self.repeat = self._check_existence('repeat')
        self.reverse = self._check_existence('reverse')
        self.room = self._check_existence('room')
        self.tshift= self._check_existence('tshift')

        self.al_transform = self._audiomentation_options()
        if self.al_transform != []:
            self.al_transform = Compose(self.al_transform)
            self.data_augmentation = True
        else:
            self.data_augmentation = False


        transforms = self._getaudiotransforms()

        if self.to_spectrogram:
            self.melbins = self._check_existence('num_mel_bins')
            self.freqm = self._check_existence('freqm') #frequency masking if freqm != 0
            self.timem = self._check_existence('timem') #time masking if timem != 0
            self.timem_p = self._check_existence('timem_p')
            
            ## dataset spectrogram mean and std, used to normalize the input
            self.norm_mean = self._check_existence('dataset_mean')
            self.norm_std = self._check_existence('dataset_std')
            ## if add noise for data augmentation
            self.noise = self._check_existence('noise')
            self.target_length = self._check_existence('target_length')
            
            self.cdo = self._check_existence('cdo')
            if self.cdo is None:
                self.cdo = False

            if self.cdo:
                self.tf_co=torch.CoarseDropout(always_apply=True,max_holes=16,min_holes=8)
            else:
                self.tf_co = None
            
            self.tfshift = self._check_existence('shift')
            if self.tfshift is None:
                self.tfshift = False
            if self.tfshift:
                self.tf_shift=torch.Affine(translate_px={'x':(0,0),'y':(0,100)})
            else:
                self.tf_shift = None

            spec_transform = self._getspectransforms()

            transforms.append(spec_transform)

        return transforms

    
    def _getaudiotransforms(self):
        """
        Use audio configuration parameters to initialize classes for audio transformation. 
        Outputs two tranform variables, one for regular audio transformation and one for 
        augmentations using albumentations

        These transformations will always load the audio. 
        :outparam audio_transform: standard transforms
        """

        waveform_loader = UidToWaveform(prefix = self.prefix, bucket=self.bucket, lib=self.use_librosa)
        transform_list = [waveform_loader]
        if self.monochannel:
            channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
            mono_tfm = ToMonophonic(reduce_fn = channel_sum)
            transform_list.append(mono_tfm)
        if self.resample_rate is not None: #16000
            downsample_tfm = ResampleAudio(resample_rate=self.resample_rate, librosa=self.use_librosa)
            transform_list.append(downsample_tfm)
        if self.trim_level is not None:
            trim_tfm = TrimBeginningEndSilence(threshold = self.trim_level, use_librosa=self.use_librosa)
            transform_list.append(trim_tfm)
        if self.clip_length is not None: #160000
            truncate_tfm = Truncate(length = self.clip_length)
            transform_list.append(truncate_tfm)

        if self.to_tensor:
            tensor_tfm = ToTensor()
            transform_list.append(tensor_tfm)
        if self.to_numpy or self.data_augmentation:
            numpy_tfm = ToNumpy()
            transform_list.append(numpy_tfm)

        
    
        # if self.clip_length is not None and self.resample_rate is not None:
        #     self.pad_size = self.clip_length*self.resample_rate
        #     pad_tfm = Pad(pad_size=self.pad_size, mode=self.pad_mode, value=self.pad_value)

        #     transforms.append(torchvision.transforms.Compose([pad_tfm]))
        

        transform = torchvision.transforms.Compose(transform_list)
        #transform_list.append(feature_tfm)

        transforms = [transform]

        if self.al_transform != []:
            transforms.append(self.al_transform)

        return transforms
    
    def _add_transform(self, transform, p, t_list):
        if p is not None:
            t_list.append(transform(p=p))
        return t_list
    
    def _audiomentation_options(self):
        t = []

        t = self._add_transform(Shift, self.tshift, t)
        
        t = self._add_transform(RoomSimulator, self.room, t)
        
        t = self._add_transform(Reverse, self.reverse, t)

        t = self._add_transform(RepeatPart, self.repeat, t)

        t = self._add_transform(TanhDistortion, self.tanh, t)

        t = self._add_transform(TimeMask, self.tmask, t)

        t = self._add_transform(TimeStretch, self.tstretch, t)

        t = self._add_transform(PolarityInversion, self.pinversion, t)

        t = self._add_transform(PitchShift, self.pshift, t)

        t = self._add_transform(Normalize, self.norm, t)

        t = self._add_transform(Mp3Compression, self.mp3, t)

        t = self._add_transform(GainTransition, self.gaint, t)

        t = self._add_transform(Gain, self.gain, t)

        t = self._add_transform(ClippingDistortion, self.clipd, t)

        t = self._add_transform(BitCrush, self.bitcrush, t)

        t = self._add_transform(BandStopFilter, self.bandstop, t)

        t = self._add_transform(Aliasing, self.alias, t)

        t = self._add_transform(AddGaussianSNR, self.gausssnr, t)

        t = self._add_transform(AddGaussianNoise, self.gauss, t)

        return t

    def _getspectransforms(self):
        '''
        Use audio configuration parameters to initialize classes for spectrogram transformation. 
        Outputs one tranform variable. Will always generate the spectrogram, and has options 
        for frequency/time masking, normalization, and adding noise

        :outparam transform: spectrogram transforms
        '''
        wav2bank = Wav2Fbank(target_length=self.target_length, melbins=self.melbins, tf_co=self.tf_co, tf_shift=self.tf_shift, override_wave=False) #override waveform so final sample does not contain the waveform - doing so because the waveforms are not the same shape
        transform_list = [wav2bank]
        if self.freqm is not None:
            freqm = FreqMaskFbank(self.freqm)
            transform_list.append(freqm)
        if self.timem is not None: 
            timem = TimeMaskFbank(self.timem, self.timem_p)
            transform_list.append(timem)
        norm = NormalizeFbank(self.norm_mean, self.norm_std)
        transform_list.append(norm)
        if self.noise:
            #TODO:
            noise = FbankNoise()
            transform_list.append(noise)
        transform = torchvision.transforms.Compose(transform_list)
        return transform

    def __getitem__(self, idx):
        '''
        Given an index, load and run transformations then return the sample dictionary

        Will run transformations in this order:
        Standard audio transformations (load audio -> reduce channels -> resample -> clip -> subtract mean) - also convert labels to tensor
        Albumentation transformations (Time shift -> speed tune -> add gauss noise -> pitch shift -> alter gain -> stretch audio)
        Spectrogram transformations (convert to spectrogram -> frequency mask -> time mask -> normalize -> add noise)

        The resulting sample dictionary contains the following info
        'uid': audio identifier
        'waveform': audio (n_channels, n_frames) or audio path
        'fbank': spectrogram (target_length, frequency_bins)
        'sample_rate': current sample rate
        'targets': labels for current file as tensor

        '''
    
        #If not doing mix-up
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if isinstance(self.data, pd.DataFrame):
            uid = self.data.index[idx] #get uid to load
        else:
            uid = self.data[idx]
    

        if self.target_labels is not None:
            targets = self.data[self.target_labels].iloc[idx].values #get target labels for given uid
            
        else:
            targets = []
      
        sample = {'uid': uid,
                  'targets':targets}
        
        sample = self.transforms[0](sample)

        if self.data_augmentation and self.transforms[1] != []:
            sample['waveform'] = self.transforms[1](samples=sample['waveform'], sample_rate = sample['sample_rate']) #audio augmentations
            if not self.to_numpy:
                sample['waveform'] = torch.from_numpy(sample['waveform']).type(torch.float32)

        #TODO: initialize mixup
        if self.mixup is not None:
            mix = Mixup()
            # if self.mixup is None:
            #     sample= mix(sample, None)

            if random.random() < self.mixup: 
                mix_sample_idx = random.randint(0, len(self.annotations_df)-1)
                mix_uid = self.annotations_df.index[mix_sample_idx]
                mix_targets = self.annotations_df[self.target_labels].iloc[mix_sample_idx].values
            
                sample2 = {
                    'uid': mix_uid,
                    'targets': mix_targets
                }
                sample2 = self.self.transforms[0](sample2) #load and perform standard transformation
                if self.data_augmentation and self.transforms[1] != []:
                    sample['waveform'] = self.transforms[1](samples=sample['waveform'], sample_rate = sample['sample_rate']) #audio augmentations
                    if not self.to_numpy:
                        sample['waveform'] = torch.from_numpy(sample['waveform']).type(torch.float32)

                sample = mix(sample, sample2)
            
            else:
                sample = mix(sample, None)

        if self.to_spectrogram:
            sample = self.transforms[-1](sample)

        if self.feature_extractor is not None:
            sample = self.feature_extractor(sample)
        
        return sample
    
    def __len__(self):
        return len(self.data)
    
