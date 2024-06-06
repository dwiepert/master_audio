"""
Download a model from a url. This is only implemented for WHISPER and NeMO

Last Modified: 04/15/2024
Author(s): Daniela Wiepert
source: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
"""
#built-in
import hashlib
import os
import urllib
import urllib.request
import warnings
from typing import List, Optional, Union

#third-party
from tqdm import tqdm
from huggingface_hub import hf_hub_url
from naip_asr.constants import *

#source: https://github.com/openai/whisper/blob/main/whisper/__init__.py

#fails frequently when used internally, so instead download like this?
def download_checkpoint_from_url(checkpoint: str, model_size: str, model_type:str, in_memory: bool = False) -> Union[bytes, str]:
    """
    Modified version of source code _download function. Includedd due to frequent errors downloading from URL when instantiating Whisper model with model type.
    Only requires checkpoints.

    
    :param checkpoint: str, path to where checkpoint should be stored (.pt for whisper, .nemo for nemo)
    :param model_size: str, size of the model to download (e.g., large, small, tiny, tiny.en)
    :param model_type: str, specify model type (e.g. whisper, nemo)
    :param in_memory: bool set to false
    :return download_target: confirmed location of the downloaded file
    """
    root = os.path.dirname(checkpoint)
    os.makedirs(root, exist_ok=True)

    if model_type == 'whisper':
        assert '.pt' in checkpoint, 'download target must be a .pt file'
        url = _WHISPER_MODELS_URL[model_size]
    else:
        raise ValueError('Can only download whisper models with this method')
    
    expected_sha256 = url.split("/")[-2]
    download_target = checkpoint

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target
