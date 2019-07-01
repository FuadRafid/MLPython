import numpy as np
from scipy.io import loadmat
import scipy.optimize as op

from Coursera.Exercise4.NNPrediction import predict_nn
from Coursera.Exercise4.RandomInitializeWeights import random_init_weights
from Coursera.Exercise4.nnCostFunction import nn_cost_function
from utils.file_utils import FileUtils

np.set_printoptions(suppress=True)
data_path = FileUtils.get_abs_path(__file__, "./data/ex4data1.mat")
mat = loadmat(data_path)
X = mat["X"]
y = mat["y"]
X = np.array(X)
y = np.array(y)

data_path = FileUtils.get_abs_path(__file__, "./data/ex4weights.mat")
mat2 = loadmat(data_path)
Theta1 = mat2['Theta1']
Theta2 = mat2['Theta2']

Theta1 = np.array(Theta1)
Theta2 = np.array(Theta2)

nn_params = np.append(Theta1.flatten(), Theta2.flatten())
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambda_value = 0
cost = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)[0]
print(cost)

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambda_value = 1
cost = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)[0]
print(cost)

initial_Theta1 = random_init_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = random_init_weights(hidden_layer_size, num_labels)


nn_params_rand = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())

lambda_value = 1
options = {'maxiter': 100}
result = op.minimize(fun=nn_cost_function, x0=nn_params_rand,
                     args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value),method='TNC',
                     jac=True, options=options)
optimal_theta = result.x

Theta1 = optimal_theta[0: hidden_layer_size * (input_layer_size + 1)]
Theta1 = np.reshape(Theta1, [hidden_layer_size, (input_layer_size + 1)])

Theta2 = optimal_theta[hidden_layer_size * (input_layer_size + 1):]
Theta2 = np.reshape(Theta2, [num_labels, (hidden_layer_size + 1)])

print(Theta1.shape)
print(Theta2.shape)

res=predict_nn(Theta1,Theta2,X)

print("Accuracy on training set with Neural Network:", np.mean((res == y)) * 100)

lambda_value = 2
options = {'maxiter': 100}
result = op.minimize(fun=nn_cost_function, x0=nn_params_rand,
                     args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value),method='TNC',
                     jac=True, options=options)
optimal_theta = result.x

Theta1 = optimal_theta[0: hidden_layer_size * (input_layer_size + 1)]
Theta1 = np.reshape(Theta1, [hidden_layer_size, (input_layer_size + 1)])

Theta2 = optimal_theta[hidden_layer_size * (input_layer_size + 1):]
Theta2 = np.reshape(Theta2, [num_labels, (hidden_layer_size + 1)])

print(Theta1.shape)
print(Theta2.shape)

res=predict_nn(Theta1,Theta2,X)

print("Accuracy on training set with Neural Network:", np.mean((res == y)) * 100)