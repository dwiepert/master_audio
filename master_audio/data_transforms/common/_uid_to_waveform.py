from master_audio.io import load_waveform_from_gcs, load_waveform_from_local, load_metadata_from_gcs, load_metadata_from_local
class UidToWaveform(object):
    '''
    Take a UID, find & load the data, add waveform and sample rate to sample
    '''
    
    def __init__(self, prefix, bucket=None, extension=None, lib=False):
        
        self.bucket = bucket
        self.prefix = prefix #either gcs_prefix or input_dir prefix
        self.cache = {}
        self.extension = extension
        self.lib = lib
        
    def __call__(self, sample):
        
        uid, targets = sample['uid'], sample['targets']
        cache = {}
        if uid not in self.cache:
            if self.bucket is not None:
                #load from google cloud storage
                wav, sr = load_waveform_from_gcs(self.bucket, self.prefix, uid, self.extension, self.lib)
                cache['waveform'] = wav 
                cache['sample_rate'] = sr
                cache['metadata'] = load_metadata_from_gcs(self.bucket, self.prefix, uid, 'json')
                self.cache[uid] = cache
            else:
                 #load local
                wav, sr = load_waveform_from_local(self.prefix, uid, self.extension, self.lib)
                cache['waveform'] = wav
                cache['sample_rate'] = sr
                cache['metadata'] = load_metadata_from_local(self.prefix, uid, 'json')
                self.cache[uid] = cache
            
        cache = self.cache[uid]
        
        sample['waveform'] = cache['waveform']
        sample['sample_rate'] = cache['sample_rate']
        sample['metadata'] = cache['metadata']
         
        return sample
    
