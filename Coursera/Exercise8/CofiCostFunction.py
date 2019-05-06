import numpy as np


def cofi_cost_function(params, Y, R, num_users, num_movies,
                       num_features, lambda_value):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    X = np.array(X)
    Theta = np.array(Theta)
    reg_theta = (lambda_value / 2) * np.sum(np.sum(Theta ** 2))
    reg_X = (lambda_value / 2) * np.sum(np.sum(X ** 2))

    J = (1 / 2) * np.sum(np.sum(((X @ Theta.T) * R - Y) ** 2)) + reg_X + reg_theta

    X_grad = ((X @ Theta.T) * R - Y) @ Theta + lambda_value * X
    Theta_grad = ((X @ Theta.T) * R - Y).T @ X + lambda_value * Theta

    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    return J, grad
