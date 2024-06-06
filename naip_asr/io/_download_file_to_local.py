from pathlib import Path
from typing import Union
import os

def download_file_to_local(gcs_path: Union[str, Path], savepath: Union[str, Path], bucket):
    """
    Download a single file to local path
    :param gcs_path: current path in gcs
    :param savepath: local path to save to
    :param bucket: GCS bucket with file
    """
    file_blob = bucket.blob(str(gcs_path))
    savepath = Path(savepath).absolute()
    if not savepath.parents[0].exists():
        os.makedirs(savepath.parents[0])
    
    file_blob.download_to_filename(str(savepath))

    return savepath

