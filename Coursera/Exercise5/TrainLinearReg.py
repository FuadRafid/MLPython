import numpy as np
import scipy.optimize as op
from Coursera.Exercise5.LinearCostFunc import linear_reg_cost


def train_linear_reg(x, y, lmbda):
    m, n = x.shape
    initial_theta = np.zeros([n, 1])

    options = {'maxiter': 200}
    result = op.minimize(fun=linear_reg_cost, x0=initial_theta, args=(x, y, lmbda), method='TNC', jac=True,
                         options=options)

    return result.x
