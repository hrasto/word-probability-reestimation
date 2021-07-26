from scipy.stats import pearsonr
import numpy as np
import random

def rsq(x, y):
    if not x.size == y.size:
        raise ValueError('x and y must be of same size. {}!={}'.format(x.size, y.size))
    fin = np.isfinite(x) & np.isfinite(y)
    r, _ = pearsonr(x[fin], y[fin])
    return r**2

def np2bow(npa):
    nz = npa.nonzero()[0]
    return list(zip(nz, npa[nz]))

def random_split(seed, iterable, maps=False, train=.5):
    random.seed(seed)
    ids = list(range(len(iterable)))
    random.shuffle(ids)
    training = ids[:int(len(iterable)*train)]
    validation = list(set(ids)-set(training))
    if maps:
        training_new = np.zeros(len(iterable)).astype(bool)
        training_new[training] = True
        training = training_new
        validation = ~training
    return training, validation 