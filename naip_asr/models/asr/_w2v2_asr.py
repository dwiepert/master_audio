"""
Set up Wav2Vec 2.0

Last modified: 05/2024
Author(s): Daniela Wiepert

Sources:
Some models to try:
https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self/tree/main
https://huggingface.co/patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram
* does this work with W2V2 instead o fAutoModelFroCTC
https://huggingface.co/patrickvonplaten/wav2vec2-large-960h-lv60-self-4-gram
https://huggingface.co/facebook/wav2vec2-large-960h
https://huggingface.co/facebook/wav2vec2-base-960h
https://huggingface.co/facebook/wav2vec2-large-robust-ft-libri-960h

"""

#IMPORTS
#built-in
from itertools import groupby
from typing import List, Union

#third-party
import librosa
import numpy as np
import soundfile as sf
import torch 

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class W2V2ForASR:
    """
    Initialize a W2V2 model for transcriptions
    """
    def __init__(self, 
                 checkpoint: str, 
                 model_size: str = None):
        """
        :param checkpoint: str, checkpoint for a pretrained model. This can either be a repo_id for a huggingface model or a full file path to a dir with the downloaded model files. For w2v2, the downloaded checkpoint should be a directory.
        :param model_size: str, specify the model size (e.g., large, base)
        """
        self._model_size = model_size
        self._checkpoint = checkpoint
        self._load_model()

    def _load_model(self):
        """
        Load a W2V2 model
        """
        self._processor = Wav2Vec2Processor.from_pretrained(self._checkpoint)
        self._model = Wav2Vec2ForCTC.from_pretrained(self._checkpoint)
        
    def _load_audio(self, 
                    audio_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio from an audio path

        :param audio_path: str, path to an audio file
        :return: audio array as an np.ndarray, sample rate
        """
        assert '.wav' in audio_path, 'Audio must be a WAV file. Please check.'
        audio, samplerate = sf.read(audio_path)
        audio = audio.T
        #audio, samplerate = librosa.load(audio_path, sr=None, mono_channel=FALSE)

        return audio, samplerate
    
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

        return audio

    
    def _get_tokens(self, 
                   audio: np.ndarray) -> np.ndarray:
        """
        Get only predicted ids

        :param audio: numpy array of the audio data
        :return predicted_ids: predicted ids for the transcription
        """

        input_values = self._processor(audio, sampling_rate=16000, return_tensors="pt").input_values
        logits = self._model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)[0]

        return predicted_ids
    
    
    def _get_timestamps(self, 
                   transcription: str,
                   predicted_ids: np.ndarray,
                   audio_size: int, 
                   ) -> List[dict]:
        """
        Get timestamps
        :param transcription: str, transcription
        :param predicted_ids: numpy array of the predicted ids that generated the transcription
        :param audio_size: len of the audio file (excluding channel dims)
        :return: List of word timestamps as a dictionary
        source: https://github.com/huggingface/transformers/issues/11307
        """

        words = [w for w in transcription.split(' ') if len(w) > 0]
        predicted_ids = predicted_ids.tolist()
        duration_sec = audio_size / 16000

        ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
        # remove entries which are just "padding" (i.e. no characers are recognized)
        ids_w_time = [i for i in ids_w_time if i[1] != self._processor.tokenizer.pad_token_id]
        # now split the ids into groups of ids where each group represents a word
        split_ids_w_time = [list(group) for k, group
                            in groupby(ids_w_time, lambda x: x[1] == self._processor.tokenizer.word_delimiter_token_id)
                            if not k]

        assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

        word_start_times = []
        word_end_times = []
        for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
            _times = [_time for _time, _id in cur_ids_w_time]
            word_start_times.append(min(_times))
            word_end_times.append(max(_times))
        
        timestamps = []
        for i in range(len(words)):
            timestamps.append({'text': words[i], 'start':word_start_times[i], 'end':word_end_times[i], 'confidence':np.nan})
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
            assert audio[-4:] == '.wav', 'Given audio file is not a WAV file.'
            audio, samplerate = self._load_audio(audio)

        assert samplerate is not None, 'If giving waveforms, must give sample rate.'
        audio = self._check_audio(audio, samplerate)
        predicted_ids = self._get_tokens(audio)
        transcription = self._processor.decode(predicted_ids).lower()
        results = {'transcription': [transcription]}

        if return_timestamps or return_pauses:
            timestamps = self._get_timestamps(transcription=transcription, predicted_ids=predicted_ids, audio_size=audio.shape[1])
        if return_timestamps:
            results['timestamps'] = [timestamps]
        if return_pauses:
            pauses = self._find_long_pauses(timestamps=timestamps, pause_s=pause_s)
            results['pauses'] = [pauses]

        return results




