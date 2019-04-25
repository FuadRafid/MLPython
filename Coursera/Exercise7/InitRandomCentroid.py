import numpy as np


def init_random_centroid(X, K):
    rand_idx = np.random.permutation(np.size(X, 0))
    centroids = X[rand_idx[0:K], :]
    return centroids
