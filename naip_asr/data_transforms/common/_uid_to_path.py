import os
from pathlib import Path
from typing import Union

from naip_asr.io import download_file_to_local

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
            temp_path = temp_path / 'waveform.wav'
            if self.bucket is None:
                self.cache[uid] = str(temp_path)
        
            else:
                save_path = self.savedir /uid
                save_path = save_path / 'waveform.wav'
                self.cache[uid] = download_file_to_local(temp_path, save_path, self.bucket)


        path = self.cache[uid]
        sample['waveform'] = str(path)


        return sample

