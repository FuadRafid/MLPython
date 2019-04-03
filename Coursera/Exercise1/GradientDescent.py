import numpy as np


def gradient_descent(x, y, theta, alpha, iterations):
    m = np.size(y, 0)
    for i in range(iterations):
        hypothesis = x @ theta
        error = hypothesis - y
        sum = np.transpose(x) @ error
        sum = sum / m
        sum = sum * alpha
        theta = theta - sum
    return theta
