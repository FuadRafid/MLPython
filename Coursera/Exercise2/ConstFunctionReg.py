import numpy as np
from Coursera.Exercise2.Sigmoid import sigmoid


def cost_function_regularized(theta, X, y, Lambda):
    m, n = X.shape
    h = X @ theta
    exp1 = -np.transpose(y) @ np.log(sigmoid(h))
    exp2 = -(np.transpose(1 - y) @ np.log(1 - sigmoid(h)))
    j_unregularized = (exp1 + exp2) / m
    regularization = (Lambda / (2 * m)) * sum(theta[1:, :] ** 2)
    j = j_unregularized + regularization

    sigmoid_x_theta = sigmoid(X.dot(theta))
    grad = ((np.transpose(X)).dot(sigmoid_x_theta - y)) / m
    grad = grad + (Lambda / m) * theta
    grad[0, 0] = ((np.transpose(X[:, 0])).dot(sigmoid_x_theta - y)) / m
    return j, grad.flatten()


def cost_function_regularized_optimization(theta, X, y):
    Lambda = 1
    m, n = X.shape
    theta = theta.reshape((n, 1))
    h = X @ theta
    exp1 = -np.transpose(y) @ np.log(sigmoid(h))
    exp2 = -(np.transpose(1. - y) @ np.log(1 - sigmoid(h)))
    j_unregularized = (exp1 + exp2) / m
    regularization = (Lambda / (2 * m)) * sum(theta[1:, :] ** 2)
    j = j_unregularized + regularization

    sigmoid_x_theta = sigmoid(X.dot(theta))
    grad = ((np.transpose(X)).dot(sigmoid_x_theta - y)) / m
    grad = grad + (Lambda / m) * theta
    grad[0, 0] = ((np.transpose(X[:, 0])).dot(sigmoid_x_theta - y)) / m
    return j, grad.flatten()
