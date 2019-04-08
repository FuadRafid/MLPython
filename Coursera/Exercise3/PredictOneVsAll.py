import numpy as np

from Coursera.Exercise3.Sigmoid import sigmoid


def predict_one_vs_all(all_theta, x):
    m, n = x.shape
    ones = np.ones([m, 1])
    x = np.hstack([ones, x])
    pred = x @ all_theta.T
    pred=sigmoid(pred)
    indices = np.argmax(pred, axis=1)
    return np.array(indices+1).reshape(m,1)