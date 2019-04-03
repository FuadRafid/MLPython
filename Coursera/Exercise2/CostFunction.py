import numpy as np
from Coursera.Exercise2.Sigmoid import sigmoid


def cost_function(theta, X, y):
    m = np.size(y, 0)
    h = X @ theta
    exp1 = -np.transpose(y) @ np.log(sigmoid(h))
    exp2 = -(np.transpose(1. - y) @ np.log(1 - sigmoid(h)))
    j = (exp1 + exp2) / m
    return j


def gradient(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((np.transpose(x)).dot(sigmoid_x_theta - y)) / m
    return grad.flatten()
