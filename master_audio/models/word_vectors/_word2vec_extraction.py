from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Word2Vec_Extract:
    """
    https://radimrehurek.com/gensim/models/word2vec.html
    https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    """

    def __init__(self, checkpoint:str):
        """
        """
        self._checkpoint = checkpoint
        self.model = Word2Vec.load_word2vec_format(self._checkpoint, binary=True)

    
    def extract(self, transcription:str):
        """
        """
        ##LEMMATIZE AND STUFF

        words = transcription.split(" ")
        vectors = []
        for w in words:
            if w in self.model.wv:
                vectors.append(self.model.wv[w])
            else:
                vectors.append(None)
        
        v_df = pd.DataFrame({'word':words, 'vector':vectors})
        return v_df
    
    def get_similarity_matrix(self, transcription:str):
        """
        """
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
                sim_matrix[i,j] = cosine_similarity(w1,w2)
        
        return v_df['word'].to_list(), sim_matrix