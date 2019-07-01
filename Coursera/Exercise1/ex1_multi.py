import numpy as np
import matplotlib.pyplot as plt
from Coursera.Exercise1.CostFunction import cost_function_j
from Coursera.Exercise1.FeatureNormalize import feature_normalize
from Coursera.Exercise1.GradientDescent import gradient_descent
from Coursera.Exercise1.NormalEqtn import normal_equation
from Coursera.Exercise1.Predict import predict
from utils.file_utils import FileUtils


def run():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex1data2.txt")
    data = np.loadtxt(data_path, delimiter=',')
    n = np.size(data, 1)
    x = data[:, range(n - 1)]
    y = data[:, n - 1]
    m = np.size(y, 0)
    x = np.reshape(x, [m, n - 1])
    y = np.reshape(y, [m, 1])
    ones = np.ones([m, 1])
    X, mu, sigma = feature_normalize(x)
    x = np.hstack([ones, x])
    X = np.hstack([ones, X])
    theta = np.zeros([n, 1])
    alpha = 0.01
    iterations = 400

    cost = cost_function_j(X, y, theta)
    print('Cost', cost)

    thetaRes,j_hist = gradient_descent(X, y, theta, alpha, iterations)
    print('Theta using gradient descent:\n', thetaRes)

    print('Price of 1650 sq ft and 3 bedroom house: ', predict([[1653, 3]], thetaRes, mu, sigma))


    plt.plot(range(400),j_hist)
    plt.xlabel("No of iterations")
    plt.ylabel("Cost")
    plt.title("Gradient Descent")
    plt.show()

    thetaRes = normal_equation(x, y)
    print('Theta using normal equation: \n', thetaRes)

    cost = thetaRes.T @ np.array([[1], [1650], [3]])
    print('Price of 1650 sq ft and 3 bedroom house: ', cost[0][0])
