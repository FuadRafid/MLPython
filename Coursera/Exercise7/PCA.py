import numpy as np


def pca(X):
    m, n = X.shape

    sigma = (1 / m) * (X.T @ X)
    U, S, V = np.linalg.svd(sigma)

    return U, S


def project_data(X, U, K):
    Z = X @ U[:, 0: K]
    return Z


def project_data_optimal_K(X, U, S):
    # Z = X @ U[:, 0: K]
    n = len(S)
    K = 0
    for i in range(n):
        test = (sum(S[0:i])) / (sum(S))
        if test >= 0.95:
            K = i
            break
    Z = X @ U[:, 0: K]
    return Z,K


def recover_data(Z, U, K):
    X_rec = Z @ U[:, 0: K].T
    return X_rec
