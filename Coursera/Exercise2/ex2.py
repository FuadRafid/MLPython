from Coursera.Exercise2.ConstFunctionReg import cost_function_regularized
from Coursera.Exercise2.PlotBoundary import plot_decision_boundary
from Coursera.Exercise2.Plotdata import plot_data,show_plot
from Coursera.Exercise2.CostFunction import cost_function, gradient
import MLBasics as ML
import numpy as np
import scipy.optimize as op

data = np.loadtxt("data/ex2data1.txt", delimiter=',')
n = np.size(data, 1)
x = data[:, range(n - 1)]
y = data[:, n - 1]
m = np.size(y, 0)
x = np.reshape(x, [m, n - 1])
y = np.reshape(y, [m, 1])
ones = np.ones([m, 1])
plot_data(x, y)
show_plot()
x = np.hstack([ones, x])
theta = np.zeros([n, 1])
cost = cost_function(theta, x, y)

print("Cost with theta [0;0;0]: ", cost)
print('Theta Result with [0;0;0]:\n', gradient(theta, x, y))

test_theta = ML.str2arr('[-24; 0.2; 0.2]')
cost = cost_function(test_theta, x, y)

print("Cost with theta [-24; 0.2; 0.2]: ", cost)
print('Theta Result with [-24; 0.2; 0.2]:\n', gradient(test_theta, x, y))

Result = op.minimize(fun=cost_function, x0=theta, args=(x, y), method='TNC', jac=gradient)
optimal_theta = Result.x
print('Optimal theta: ',optimal_theta)

optimal_theta=np.resize(optimal_theta,[optimal_theta.shape[0],1])
res = x @ optimal_theta
res[res >= 0.5] =1
res[res <= 0.5] =0
print('Accuracy: ',np.mean(((res==y).flatten()))*100)
plot_decision_boundary(optimal_theta,x,y)