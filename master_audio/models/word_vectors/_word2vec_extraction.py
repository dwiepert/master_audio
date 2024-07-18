from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm

class Word2Vec_Extract:
    """
    https://radimrehurek.com/gensim/models/word2vec.html
    https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    """

    def __init__(self, checkpoint:str):
        """
        """
        self._checkpoint = checkpoint
        self.model = KeyedVectors.load_word2vec_format(self._checkpoint, binary=True)

    
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
        """
        """
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