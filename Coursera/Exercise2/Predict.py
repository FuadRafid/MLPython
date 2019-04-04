from Coursera.Exercise2.Sigmoid import sigmoid
import numpy as np


def predict(theta, x):
    theta=theta.flatten()
    theta = np.resize(theta, [theta.shape[0], 1])
    res = sigmoid(x @ theta)
    res[res >= 0.5] = 1
    res[res < 0.5] = 0
    return res
