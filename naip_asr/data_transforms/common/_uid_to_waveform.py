from naip_asr.io import load_waveform_from_gcs, load_waveform_from_local
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
        
        if uid not in self.cache:
            if self.bucket is not None:
                #load from google cloud storage
                self.cache[uid] = load_waveform_from_gcs(self.bucket, self.prefix, uid, self.extension, self.lib)
            else:
                 #load local
                 self.cache[uid] = load_waveform_from_local(self.prefix, uid, self.extension, self.lib)

            
        waveform, sample_rate = self.cache[uid]
        
        sample['waveform'] = waveform
        sample['sample_rate'] = sample_rate
         
        return sample
    
