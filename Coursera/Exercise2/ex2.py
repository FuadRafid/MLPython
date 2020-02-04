import numpy as np
import scipy.optimize as op

from Coursera.Exercise2.CostFunction import cost_function
from Coursera.Exercise2.PlotBoundary import plot_decision_boundary
from Coursera.Exercise2.Predict import predict
from utils.String2NpArray import str2arr
from utils.file_utils import FileUtils


def run():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex2data1.txt")
    data = np.loadtxt(data_path, delimiter=',')
    n = np.size(data, 1)
    x = data[:, range(n - 1)]
    y = data[:, n - 1]
    m = np.size(y, 0)
    x = np.reshape(x, [m, n - 1])
    y = np.reshape(y, [m, 1])
    ones = np.ones([m, 1])
    x = np.hstack([ones, x])
    theta = np.zeros([n, 1])
    cost,grad = cost_function(theta, x, y)

    print("Cost with theta [0;0;0]: ", cost)
    print('Theta Result with [0;0;0]:\n', grad)

    test_theta = str2arr('[-24; 0.2; 0.2]')
    cost,grad = cost_function(test_theta, x, y)

    print("Cost with theta [-24; 0.2; 0.2]: ", cost)
    print('Theta Result with [-24; 0.2; 0.2]:\n', grad)

    Result = op.minimize(fun=cost_function, x0=theta, args=(x, y), method='TNC', jac=True)
    optimal_theta = Result.x
    print('Optimal theta: ', optimal_theta)

    res=predict(optimal_theta,x)
    print("Accuracy:", np.mean(((res == y).flatten())) * 100)
    plot_decision_boundary(optimal_theta, x, y)
