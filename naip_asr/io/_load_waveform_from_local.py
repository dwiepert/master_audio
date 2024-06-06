#third party
import librosa
import torch
import torchaudio
import torch.nn.functional

def load_waveform_from_local(input_dir, uid, extension = None, lib=False):
    """
    :param input_directory: directory where data is stored locally
    :param uid: audio identifier
    :param extension: audio type (default, None)
    :param lib: boolean indicating to load with librosa rather than torchaudio

    :return: loaded audio waveform as tensor
    """
    
    if extension is None:
        extension = 'wav'
        
    waveform_path = f'{input_dir}/{uid}/waveform.{extension}'
    
    if not lib:
        waveform, sr = torchaudio.load(waveform_path, format = extension)
    else:
        waveform, sr = librosa.load(waveform_path, mono=False, sr=None)
        waveform = torch.from_numpy(waveform)
        if len(waveform.shape) == 1:
           waveform = waveform.unsqueeze(0)
    
    return waveform, sr