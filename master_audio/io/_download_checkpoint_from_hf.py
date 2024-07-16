"""
Download a checkpoint using hugging face.

Last Modified: 04/15/2024
Author(s): Daniela Wiepert
Source:
"""
import os
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download, snapshot_download

from master_audio.constants import *

def download_checkpoint_from_hf(checkpoint: Union[str, Path],  model_type: str = None, model_size: str = None,
                                repo_id: Optional[str] = None, filename: Optional[str]=None, subfolder: Optional[str] = None):
    """
    Download checkpoint from hugging face

    :param checkpoint: str, Path, directory where the checkpoint file(s) should be saved. Must be full file path to where to save checkpoint
    :param model_size: str, size of the model to download (e.g., large, small, tiny, tiny.en)
    :param model_type: str, specify model type (e.g. whisper)
    :param repo_id: str, repo_id in hugging face
    :param filename: optional filename if downloading a single file instead of directory
    :param subfolder: str, optional, specify if there is a file in a subdirectory of the hugging face repo
    """
    checkpoint = Path(checkpoint).absolute()

    if not checkpoint.exists():
        os.makedirs(checkpoint) 
    
    if model_type == 'whisper' or model_type=='ssast':
        raise NotImplementedError
        # assert model_size in _WHISPER_MODELS,  f'model_size {model_size} not implemented for Whisper.'
        # repo_id = _WHISPER_MODELS[model_size]
        #'Whisper not compatible with hugging face checkpoints, download from url instead.'
    elif model_type == 'w2v2':
        assert model_size in _W2V2_MODELS, f'model_size {model_size} not implemented for Wav2vec2.'
        repo_id = _W2V2_MODELS[model_size]
    else:
        assert repo_id is not None, 'If model_type not specified, repo_id must be specified.'
    
    if filename is not None:
        hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=checkpoint)
    else:
        snapshot_download(repo_id=repo_id, local_dir=checkpoint)
