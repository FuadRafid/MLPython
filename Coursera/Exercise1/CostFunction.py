import numpy as np


def cost_function_j(x, y, theta):
    m = np.size(x, 0)
    predictions = x @ theta
    sqr_error = (predictions - y) ** 2
    j = 1 / (2 * m) * sum(sqr_error)
    return j[0]
