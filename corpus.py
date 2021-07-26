#import scipy
#import scipy.sparse
import numpy as np
import pickle

class WordCount(object):
    def __init__(self, corpus, vocab: list):
        """Constructor

        Args:
            corpus (np.ndarray or matrix): Word counts either as a vector or word-document matrix.
            vocab (list): List of words that the indices in corpus correspond to. 

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        if len(corpus.shape) == 2:
            self.corpus = corpus.sum(axis=1)
            if not len(vocab) == len(self.corpus):
                raise ValueError('Vocab size must equal the first dimension of the word-document matrix')
        elif len(corpus.shape) == 1:
            self.corpus = corpus
        else:
            raise ValueError('Corpus must have either 1 (word counts) or 2 dimensions (word-document matrix).')

        #self.docs = scipy.sparse.load_npz(corpus_path)
        #with open(vocab_path, 'rb') as f:
        #vocab = pickle.load(f)
        self.vocab = np.asarray(vocab)
        self.freq = np.asarray(self.corpus).flatten()
        self.freq_million = self.freq / (self.freq.sum()/1e6)
        self.freq_million_log = np.log10(self.freq_million)

