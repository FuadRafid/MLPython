import numpy as np

from Coursera.Exercise5.LinearCostFunc import linear_reg_cost
from Coursera.Exercise5.TrainLinearReg import train_linear_reg


def learning_curve(x, y, xval, yval, lambda_value):
    m, n = x.shape
    error_train = np.zeros([m])
    error_val = np.zeros([m])

    for i in range(1,m+1):
        x_t = x[0:i, :]
        y_t = y[0:i, :]
        theta = train_linear_reg(x_t, y_t, lambda_value)
        error_train[i-1] = linear_reg_cost(theta,x_t, y_t, 0)[0]
        error_val[i-1] = linear_reg_cost(theta,xval, yval, 0)[0]

    return error_train,error_val
