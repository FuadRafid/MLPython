import numpy as np


def estimate_gaussian(x):
    m, n = x.shape

    mu = np.mean(x,0)
    sigma = np.var(x,0)

    return mu, sigma
