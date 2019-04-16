import numpy as np


def linear_reg_cost(theta,x, y, lmbda):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    predictions = x @ theta
    sqr_error = (predictions - y) ** 2
    j_unregularized = 1 / (2 * m) * sum(sqr_error)
    regularization = (lmbda / (2 * m)) * sum(theta[1:, :] ** 2)
    j = j_unregularized + regularization

    grad = (x.T @ (predictions - y)) / m
    grad = grad + (lmbda / m) * theta
    grad[0, 0] = ((np.transpose(x[:, 0])).dot(predictions - y)) / m

    return j[0], grad.flatten()
