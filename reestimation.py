import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
from gensim.models import LdaMulticore
import os

logf2logp = lambda x: np.log10((10**x)/(10**x).sum())

def reestimate(beta, coef):
    """Compute reestimated word probabilities given coefficients coef.

    Args:
        coef (np.ndarray): Coefficients of shape (n_components, n_models).

    Returns:
        np.ndarray: Matrix of shape (n_words, n_models).
    """
    res = beta.dot(coef)
    res = 10**res
    res = np.log10(res/res.sum(axis=0))
    
    return res

def fit(beta, noisy):
    """Fit components to given noisy data

    Args:
        noisy (np.ndarray): Noisy representation(s) to fit. Matrix of shape (n_words, n_models).
    Returns:
        np.ndarray: Matrix of shape (n_components, n_models).
    """
    if noisy.shape[0] != beta.shape[0]:
        raise ValueError('First dimension of noisy must match the first dimension of beta. {}!={}'.format(beta.shape[0], noisy.shape[0]))
    # leave out inaccurate responses (nans)
    fin = np.isfinite(noisy.sum(axis=1)) if len(noisy.shape) > 1 else np.isfinite(noisy)
    # fit linear model
    lm = LinearRegression()
    lm.fit(beta[fin, :], noisy[fin])

    return lm.coef_.T

class LDA(object):
    def __init__(self, ):
        self.n_topics = []
        self.models = {}

    def load_models(self, models_dir: str, n_topics: list):
        """Loads LDA models train by train_models from given directory. 

        Args:
            models_dir (str): Directory with the trained models. 
            n_topics (list): List of the values for the hyper-parameter number of topics. 
        """
        self.n_topics = []
        self.models = {}
        for n_components in n_topics:
            modelfile = "{}/LDA-n{}/LDA-n{}".format(models_dir, n_components, n_components)
            try:
                self.models[n_components] = LdaMulticore.load(modelfile)
                self.n_topics.append(n_components)
            except Exception:
                warnings.warn("Did not find trained LDA model at {}.".format(modelfile))

    def train_models(self, models_dir: str, docs, words: list, n_topics: list):
        if not docs.shape[1] == len(words):
            raise ValueError('Second dimension of docs must match the length of words. {}!={}'.format(docs.shape[1], len(words)))
        docs = docs.tocsr()
        gensim_input = []
        for i in range(docs.shape[0]):
            row = docs.getrow(i).toarray().flatten()
            nonzero = row.nonzero()[0]
            gensim_input.append(list(zip(nonzero, row[nonzero])))
            
        if not os.path.isdir(models_dir):
            os.mkdir(models_dir)

        id2word=dict(zip(range(len(words)), words))
        for topic_num in n_topics:
            print(topic_num)
            modeldir = "{}/LDA-n{}".format(models_dir, topic_num)
            if not os.path.isdir(modeldir):
                os.mkdir(modeldir)
            topic_model = LdaMulticore(gensim_input, num_topics=topic_num, id2word=id2word, iterations=20, random_state=1)
            topic_model.save(os.path.join(modeldir, "LDA-n{}".format(topic_num)))

    def get_beta(self, n_components=20):
        """Returns the estimated parameter beta for topic model with given number of topics.

        Args:
            n_components (int, optional): Number of components of the desired topic model. Defaults to 20.
            logscale (bool, optional): Whether or not to return log-scaled values. Defaults to True.

        Raises:
            ValueError: Raised if the model with given n_components is not loaded. 

        Returns:
            np.ndarray: Matrix of shape (n_words, n_components).
        """
        if n_components not in self.n_topics:
            raise ValueError('Specified n_components={} not available in [{}]'.format(n_components, ','.join(list(self.models.keys()))))
        res = self.models[n_components].get_topics()
        res /= res.min()
        res = np.log10(res).T
        
        return res


