import numpy as np


def predict(x, theta, mu, sigma):
    x = (x - mu) / sigma
    m = np.size(x, 0)
    ones = np.ones([m, 1])
    x = np.hstack([ones, x])
    res = theta.T @ np.array(x).T
    return res[0][0]
