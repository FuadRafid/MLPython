from Coursera.Exercise2.MapFeature import map_features
from Coursera.Exercise2.PlotBoundary import plot_decision_boundary
from Coursera.Exercise2.ConstFunctionReg import cost_function_regularized
import MLBasics as ML
import numpy as np
import scipy.optimize as op

from Coursera.Exercise2.Predict import predict
from Coursera.Exercise2.Sigmoid import sigmoid

np.set_printoptions(suppress=True)
data = np.loadtxt("data/ex2data2.txt", delimiter=',')
n = np.size(data, 1)
x = data[:, range(n - 1)]
y = data[:, n - 1]
m = np.size(y, 0)
x = np.reshape(x, [m, n - 1])
y = np.reshape(y, [m, 1])
ones = np.ones([m, 1])
x = map_features(x[:, 0], x[:, 1])
theta = np.zeros([np.size(x, 1), 1])
cost, theta_res = cost_function_regularized(theta, x, y, 1)

print("Cost with theta [0;0;0]: ", cost[0][0])
print('Theta Result with [0;0;0]:\n', theta_res)


test_theta = np.ones([np.size(x, 1), 1])
cost, theta_res = cost_function_regularized(test_theta, x, y, 10)

print("Cost with test theta : ", cost[0][0])
print('Theta Result with test theta:\n', theta_res)


test_theta = np.zeros([np.size(x, 1), 1])
options = {'maxiter': 400}
Result = op.minimize(fun=cost_function_regularized, x0=test_theta, args=(x, y ,1), method='TNC', jac=True,
                     options=options)
optimal_theta = Result.x
print('Optimal theta: ', optimal_theta)

res=predict(optimal_theta,x)
print(np.mean(((res == y).flatten())) * 100)
plot_decision_boundary(optimal_theta, x, y)
