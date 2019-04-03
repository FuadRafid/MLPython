import numpy as np


def cost_function_j(x, y, theta):
    m = np.size(x, 0)
    predictions = x @ theta
    sqr_error = (predictions - y) ** 2
    j = 1 / (2 * m) * sum(sqr_error)
    return j


def gradient_descent(x, y, theta, alpha, iter):
    m = np.size(y, 0)
    for i in range(iter):
        hypothesis = x @ theta
        error = hypothesis - y
        sum = np.transpose(x) * error
        sum = sum / m
        sum = sum * alpha
        theta = theta - sum
    return theta
