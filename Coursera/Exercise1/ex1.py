from utils.file_utils import FileUtils
import matplotlib.pyplot as plt
import numpy as np

import Coursera.Exercise1.plotData as pltData
import MLBasics as ML
from Coursera.Exercise1.CostFunction import cost_function_j
from Coursera.Exercise1.GradientDescent import gradient_descent
from mpl_toolkits.mplot3d import Axes3D




def run():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex1data1.txt")
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
    alpha = 0.01
    iterations = 1500
    cost = cost_function_j(x, y, theta)
    print('Cost', cost)

    thetaRes,j_hist = gradient_descent(x, y, theta, alpha, iterations)
    print(thetaRes)

    cost = cost_function_j(x, y, ML.str2arr('[-1;2]'))
    print(cost)

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.vstack([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = cost_function_j(x, y, t)

    pltData.plot_data()
    plt.plot(x[:, 1], x @ thetaRes, '-', color='red')

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)

    ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

    ax2.plot_surface(theta0_vals, theta1_vals, np.transpose(J_vals))
    plt.show()
