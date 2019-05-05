import numpy as np


def multivariate_gaussian(X, mu, sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.
    """
    k = len(mu)

    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma2) ** 0.5)) * np.exp(
        -0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    return p
