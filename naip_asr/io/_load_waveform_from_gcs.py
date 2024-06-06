#built-in
import io

#third party
import librosa
import torch
import torchaudio
import torch.nn.functional


def load_waveform_from_gcs(bucket, gcs_prefix, uid, extension = None, lib=False):
    """
    load audio from google cloud storage
    :param bucket: gcs bucket object
    :param gcs_prefix: prefix leading to object in gcs bucket
    :param uid: audio identifier
    :param extension: audio type (default, None)
    :param lib: boolean indicating to load with librosa rather than torchaudio
    :return:  loaded audio waveform as tensor 
    """
   
    if extension is None:
        extension = 'wav'
        
    gcs_waveform_path = f'{gcs_prefix}/{uid}/waveform.{extension}'
    
    blob = bucket.blob(gcs_waveform_path)
    wave_string = blob.download_as_string()
    wave_bytes = io.BytesIO(wave_string)
    if not lib:
        waveform, sr = torchaudio.load(wave_bytes, format = extension)
    else:
        waveform, sr = librosa.load(wave_bytes, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
           waveform = waveform.unsqueeze(0)
    
    return waveform, sr