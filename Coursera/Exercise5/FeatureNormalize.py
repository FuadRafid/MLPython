import numpy as np


def feature_normalize(x):
    X_norm = x
    mu = np.zeros([np.size(x, 1)])
    sigma = np.zeros([np.size(x, 1)])
    m = np.size(x, 1)

    for i in range(m):
        mu[i] = np.mean(x[:, i])
        sigma[i] = np.std(x[:, i], ddof=1)


    X_norm = X_norm - mu
    X_norm = X_norm / sigma
    return X_norm, mu, sigma
