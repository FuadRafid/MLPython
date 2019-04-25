import numpy as np


def compute_centroids(X, idx, K):
    m, n = X.shape
    count = np.zeros([K])

    centroids = np.zeros([K, n])

    for i in range(m):
        idxVal = idx[i]
        centroids[idxVal, :] += X[i, :]
        count[idxVal] += 1

    for i in range(K):
        centroids[i, :] /= count[i]

    return centroids
