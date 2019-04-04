import numpy as np

from Coursera.Exercise1.CostFunction import cost_function_j


def gradient_descent(x, y, theta, alpha, iterations):
    m = np.size(y, 0)
    j_hist=[]
    for i in range(iterations):
        hypothesis = x @ theta
        error = hypothesis - y
        sum = np.transpose(x) @ error
        sum = sum / m
        sum = sum * alpha
        theta = theta - sum
        j_hist.append(cost_function_j(x,y,theta))
    return theta,j_hist
