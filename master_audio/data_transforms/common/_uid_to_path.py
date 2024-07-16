import os
from pathlib import Path
from typing import Union
from tempfile import TemporaryDirectory

from master_audio.io import download_file_to_local, load_metadata_from_gcs, load_metadata_from_local

class UidToPath(object):
    """
    Take a UID and convert to an absolute path. Download to local computer if necessary.
    """

    def __init__(self, prefix, savedir:Union[Path,str] = None, bucket=None):
        self.prefix = prefix
        self.savedir = savedir
        self.bucket = bucket
        if self.bucket is not None:
            assert self.savedir is not None, 'must have a directory to save to if downloading from bucket'
            self.savedir = Path(self.savedir).absolute()
            if not self.savedir.exists():
                os.makedirs(savedir)
        self.cache = {}

    def __call__(self, sample):
        uid, targets = sample['uid'], sample['targets']

        if uid not in self.cache:
            temp_path = self.prefix / uid
            temp_wav_path = temp_path / 'waveform.wav'
            
            cache = {}

            if self.bucket is None:
                cache['waveform'] = str(temp_wav_path)
                cache['metadata'] = load_metadata_from_local(self.prefix, uid, 'json')
                self.cache[uid] = cache
        
            else:

                with TemporaryDirectory() as tempdir:
                    save_path = Path(tempdir) /uid
                    save_path = save_path / 'waveform.wav'
                    cache['waveform'] = download_file_to_local(temp_wav_path, save_path, self.bucket)
                
                cache['metadata'] = load_metadata_from_gcs(self.bucket, self.prefix, uid, 'json')
                self.cache[uid] = cache


        cache = self.cache[uid]
        sample['waveform'] = cache['waveform']
        sample['metadata'] = cache['metadata']


        return sample

