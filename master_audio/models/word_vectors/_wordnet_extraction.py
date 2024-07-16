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
        self._stopwords = nltk.corpus.stopwords.words("english")
        self._lemmatization = nltk.stem.wordnet.WordNetLemmatizer()

    def get_similarity_matrix(self, transcription, remove_stopwords: bool=False, lemmatize: bool = False):
        """
        """
        #TODO, process transcription

        processed = word_tokenize(transcription)

        if remove_stopwords:
            processed = [w for w in processed if w not in self._stopwords]
        
        if lemmatize:
            processed = [self._lemmatization.lemmatize(w) for w in processed]
        
        tl = len(processed)
        sim_matrix = np.empty((tl,tl))
        for i in range(tl):
            for j in range(tl):
                w1 = wn.synsets(processed[i])[0] #What if empty? TODO
                w2 =wn.synsets(processed[j])[0]
                sim_matrix[i,j] = w1.wup_similarity(w2)
        
        return processed, sim_matrix


