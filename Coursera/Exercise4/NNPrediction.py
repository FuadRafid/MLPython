import numpy as np
from Coursera.Exercise4.Sigmoid import sigmoid


def predict_nn(Theta1, Theta2, X):
    m, n = X.shape
    ones = np.ones([m, 1])
    X = np.hstack([ones, X])
    a2 = sigmoid(X @ Theta1.T)
    m2 = np.size(a2, 0)
    a2 = np.hstack([np.ones([m2, 1]),a2])
    pred = sigmoid(a2 @ Theta2.T)
    indices = np.argmax(pred, axis=1)
    return np.array(indices + 1).reshape(m, 1)