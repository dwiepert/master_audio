from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

class WordNet:
    """
    """
    def __init__(self):
        """
        """
        #wn = nltk.download('wordnet')

    def get_similarity_matrix(self, transcription, remove_stopwords: bool=False, lemmatize: bool = False):
        """
        """
        #TODO, process transcription

        processed = transcription.split(" ") 
        
        tl = len(processed)
        sim_matrix = np.empty((tl,tl))
        for i in range(tl):
            for j in range(tl):

                w1 = wn.synsets(processed[i])
                w2 =wn.synsets(processed[j])

                #check for empty:
                if w1 != [] and w2 != []:
                    w1 = w1[0]
                    w2 = w2[0]
                    sim_matrix[i,j] = w1.wup_similarity(w2)
                else:
                    sim_matrix[i,j] = np.nan
        
        return processed, sim_matrix


