import numpy as np


def find_closest_centroids(X, centroids):
    idx = np.zeros([np.size(X, 0), 1], dtype=int)
    m = np.size(X, 0)

    for i in range(m):
        dist = centroids - X[i, :]
        dist = np.power(dist, 2)
        dist = np.sum(dist, axis=1)
        idx[i] = np.argmin(dist)

    return idx
