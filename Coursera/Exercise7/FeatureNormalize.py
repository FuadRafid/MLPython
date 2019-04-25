import numpy as np


def feature_normalize(x):
    X_norm = x
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0,ddof=1)
    X_norm = X_norm - mu
    X_norm = X_norm / sigma
    return X_norm, mu, sigma
