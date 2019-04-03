import numpy as np


def normal_equation(x, y):
    theta = np.linalg.pinv(x.T @ x) @ x.T @ y
    return theta
