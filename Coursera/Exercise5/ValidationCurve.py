import numpy as np

from Coursera.Exercise5.LinearCostFunc import linear_reg_cost
from Coursera.Exercise5.TrainLinearReg import train_linear_reg


def validation_curve(x, y, xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):
        theta = train_linear_reg(x, y, lambda_vec[i])
        error_train[i] = linear_reg_cost(theta, x, y, 0)[0]
        error_val[i] = linear_reg_cost(theta, xval, yval, 0)[0]
    return lambda_vec,error_train, error_val
