import fasttext
import pandas as pd
from pathlib import Path
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        self._checkpoint = checkpoint
        self.model_type = model_type
        self.model = fasttext.load_model(self._checkpoint)

    def extract(self, transcription:str):
        """
        """
        #PREPROCESS TRANSCRIPTION? 
        #TODO

        words = transcription.split(" ") 
        vectors = []
        for w in words:
            if w in self.model:
                vectors.append(self.model[w])
            else:
                vectors.append(None)

        v_df = pd.DataFrame({'word':words, 'vector':vectors})
        
        return v_df
    
    def get_similarity_matrix(self, transcription:str):
        v_df = self.extract(transcription)

        vectors = v_df['vectors'].to_list()

        tl = len(vectors)
        sim_matrix = np.empty((tl,tl))

        for i in range(tl):
            for j in range(tl):
                w1 = vectors[i]
                w2 = vectors[j]
                if w1 is None or w2 is None:
                    sim_matrix[i,j] = np.nan
                else:
                    sim_matrix[i,j] = cosine_similarity(w1,w2)
        
        return v_df['word'].to_list(), sim_matrix