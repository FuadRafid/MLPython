import numpy as np

from Coursera.Exercise4.Sigmoid import sigmoid
from Coursera.Exercise4.SigmoidGradient import sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value):
    m, n = X.shape
    Theta1 = nn_params[0: hidden_layer_size * (input_layer_size + 1)]
    Theta1 = np.reshape(Theta1, [hidden_layer_size, (input_layer_size + 1)])

    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = np.reshape(Theta2, [num_labels, (hidden_layer_size + 1)])

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    y_t = np.zeros([m, num_labels])
    for i in range(m):
        idx = y[i] - 1
        y_t[i, idx] = 1

    x = np.hstack([np.ones([m, 1]), X])
    a1 = x
    z2 = x @ Theta1.T

    a2 = sigmoid(z2)

    m2 = np.size(a2, 0)
    a2 = np.hstack([np.ones([m2, 1]), a2])

    z3 = a2 @ Theta2.T

    h = sigmoid(z3)
    a3 = h

    exp1 = -(y_t) * np.log((h))
    exp2 = -(1. - y_t) * np.log(1 - (h))
    j_unregularized = sum(sum(exp1 + exp2)) / m
    regularization = (lambda_value / (2 * m)) * (sum(sum(Theta1[:, 1:] ** 2)) + sum(sum(Theta2[:, 1:] ** 2)))
    j = j_unregularized + regularization

    d3 = a3 - y_t
    d2 = d3 @ Theta2
    d2 = d2 * sigmoid_gradient(np.hstack([np.ones([m, 1]), z2]))
    d2 = d2[:, 1:]

    Theta1_grad = Theta1_grad + d2.T @ a1

    Theta2_grad = Theta2_grad + d3.T @ a2

    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    reg_theta_1 = (lambda_value / m) * Theta1[:, 1:]
    reg_theta_2 = (lambda_value / m) * Theta2[:, 1:]

    reg_theta_1 = np.hstack([np.zeros([np.size(Theta1, 0), 1]), reg_theta_1])
    reg_theta_2 = np.hstack([np.zeros([np.size(Theta2, 0), 1]), reg_theta_2])

    Theta1_grad = Theta1_grad + reg_theta_1
    Theta2_grad = Theta2_grad + reg_theta_2

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())

    return j, grad
