import numpy as np
from Coursera.Exercise2.Sigmoid import sigmoid


def cost_function(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    h = X @ theta
    exp1 = -np.transpose(y) @ np.log(sigmoid(h))
    exp2 = -(np.transpose(1. - y) @ np.log(1 - sigmoid(h)))
    j = (exp1 + exp2) / m
    y = y.reshape((m, 1))
    sigmoid_x_theta = sigmoid(X.dot(theta))
    grad = ((np.transpose(X)).dot(sigmoid_x_theta - y)) / m

    return j[0][0],grad.flatten()
