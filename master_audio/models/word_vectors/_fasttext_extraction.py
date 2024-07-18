import fasttext
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm

class FastText:
    """
    https://fasttext.cc/docs/en/english-vectors.html
    """
    def __init__(self, 
                 checkpoint: str,
                 model_type: str):
        """
        *Note that all files must already be saved locally
        :param checkpoint: str, checkpoint for a trained model
        
        """
        self._checkpoint = str(checkpoint)
        assert '.bin' in self._checkpoint, 'Must give a .bin file to load fasttext'
        self.model_type = model_type

        self.model = fasttext.load_model(self._checkpoint)

    def extract(self, transcription:str):
        """
        """

        words = transcription.split(" ") 
        vectors = None
        for i in range(len(words)):
            w = words[i]
            if w in self.model:
               temp = self.model[w].reshape(-1,1)
            else:
                temp = np.empty((300,1))
                temp[:,:] = np.nan
            
            if vectors is None:
                vectors = temp
            else:
                vectors = np.concatenate((vectors, temp), axis=1)

        return words, vectors
    
    def get_similarity_matrix(self, transcription:str):
        words, vectors = self.extract(transcription)
        tl = len(words)
        sim_matrix = np.empty((tl,tl))
        temp = np.empty((300,1))
        temp[:,:] = np.nan

        for i in range(tl):
            for j in range(tl):
                w1 = vectors[:,i]
                w2 = vectors[:,j]
                if (w1 == temp).all() or (w2 == temp).all():
                    sim_matrix[i,j] = np.nan
                else:
                    sim_matrix[i,j] = np.dot(w1,w2)/(norm(w1)*norm(w2))
        
        return words, sim_matrix