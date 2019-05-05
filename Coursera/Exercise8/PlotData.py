import numpy as np
from matplotlib import pyplot

from Coursera.Exercise8.MutlivariateGaussian import multivariate_gaussian


def visualize_fit(X, mu, sigma2):

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)

    pyplot.scatter(X[:, 0], X[:, 1],color='green',marker='x')

    if np.all(abs(Z) != np.inf):
        pyplot.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), zorder=100)

