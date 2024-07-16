"""
Set up Whisper

Last modified: 07/2024
Author(s): Daniela Wiepert

Sources: 
https://github.com/openai/whisper
https://github.com/linto-ai/whisper-timestamped

"""
#IMPORTS
#built-in
import os
import string
from itertools import groupby
from typing import List, Union

#third-party
#import whisper
import librosa
import numpy as np
import soundfile as sf
import torch

#import whisper
import whisper_timestamped as whisper

#local


class WhisperForASR:
    """
    Initialize a Whisper model
    """
    def __init__(self, 
                 checkpoint: str, 
                 model_size: str = None):
        """
        :param checkpoint: str, TODO: checkpoint for a pretrained model. This can either be a repo_id for a huggingface model or a full file path to a downloaded model files. For whisper, the downloaded checkpoint should be a .pt file.
        :param model_size: str, specify the model type (e.g., large, base)
        """
        self._model_size = model_size
        self._checkpoint = checkpoint
        self._load_model()

    def _load_model(self):
        """
        Load whisper model
        """
        #try:
        self._model = whisper.load_model(self._checkpoint, device="cpu") #takes to long, instead of try except, check if checkpoint exists, and then try model type instead? TODO
        
    def _check_audio(self, 
                     audio: np.ndarray,
                     samplerate: int)-> np.ndarray:
        """
        check audio is in the proper format (correct shape, correct sample rate, monochannel)
        :param audio: numpy array of the audio data
        :param samplerate: current sample rate of the audio
        :return audio: numpy array of the audio data after processing
        """
        #TODO: check shape is right
        if audio.ndim == 2 and audio.shape[1] <= 2:
            audio = audio.T #TODO: check
        if samplerate != 16000:
            audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)
        
        if (audio.ndim == 2 and audio.shape[0] == 2): 
            audio = librosa.to_mono(audio)
    
        if audio.ndim != 2:
            audio = np.expand_dims(audio, axis=0) #TODO: check

        audio = audio.astype(np.float32)

        return audio
    
    def _get_timestamps(self, result:dict) -> List[dict]:
        """
        Get timestamps
        :param result: output result from whisper model
        :return: List of word timestamps as a dictionary
        source: https://github.com/linto-ai/whisper-timestamped

        """
        segments = result["segments"]
        if segments == []:
            return []
        timestamps = segments[0]["words"]
        for i in range(len(timestamps)):
            t = timestamps[i]
            text = t['text']
            text = text.lower().strip()
            text = text.translate(str.maketrans('','',string.punctuation))
            t['text'] = text
            timestamps[i] = t
        
        return timestamps
     
    
    def _find_long_pauses(self,
                         timestamps: List[dict],
                        pause_s: float = 0.1) -> List[dict]:
        """
        Find pauses longer than given threshold
        :param timestamps: List[dict], feed already found timestamps or find from scratch
        :param pause_s: threshold for long pause in seconds
        :return: list of long pauses
        """
        
        pauses = []
        for i in range(len(timestamps)):
            if i < (len(timestamps) - 1):
                curr_end = timestamps[i]['end']
                next_start = timestamps[i+1]['start']
                diff_s = abs(next_start - curr_end)
                if diff_s >= pause_s:
                    pauses.append({'diff_s': diff_s, 'pause_start': curr_end, 'pause_end': next_start, 'start_word':timestamps[i]['text'], 'end_word':timestamps[i+1]['text']})

        return pauses
    
    
    
    def transcribe(self, 
                   audio: Union[str, np.ndarray],
                   samplerate: int = None, 
                   return_timestamps: bool = True,
                   return_pauses: bool = True,
                   pause_s: float = 0.1) -> dict:
        """
        Transcribe a single audio file
        :param audio: either a file path to a .wav audio file as a string or an already loaded waveform.
        :param return_timestamps: boolean indicating whether to return timestamps
        :param return_pauses: boolean indicating whether to return pauses
        :return results: dictionary with transcription and optionally timestamps and pauses
        """
        if isinstance(audio, str):
            audio = whisper.load_audio(audio)
        else:
            assert samplerate is not None, 'If giving audio as waveform, original sampling rate must also be given.'
            audio = self._check_audio(audio, samplerate)
        
        all = whisper.transcribe(self._model, audio, language="en")
        transcription = all["text"].lower().strip()
        transcription = transcription.translate(str.maketrans('','',string.punctuation))

        results = {'transcription': [transcription]}

        if return_timestamps or return_pauses:
            timestamps = self._get_timestamps(all)
        if return_timestamps:
            results['timestamp'] = [timestamps]
        if return_pauses:
            pauses = self._find_long_pauses(timestamps=timestamps, pause_s=pause_s)
            results['pause'] = [pauses]
        
        return results




