import numpy as np
import pandas as pd
import scipy.optimize as op
from scipy.io import loadmat

from Coursera.Exercise8.CofiCostFunction import cofi_cost_function
from Coursera.Exercise8.NormalizeRatings import normalize_ratings
from utils.file_utils import FileUtils


def run():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex8_movies.mat")
    mat3 = loadmat(data_path)

    data_path = FileUtils.get_abs_path(__file__, "./data/ex8_movieParams.mat")
    mat4 = loadmat(data_path)

    Y = mat3["Y"]  # 1682 X 943 matrix, containing ratings (1-5) of 1682 movies on 943 user
    R = mat3["R"]  # 1682 X 943 matrix, where R(i,j) = 1 if and only if user j give rating to movie i
    X = mat4["X"]  # 1682 X 10 matrix , num_movies X num_features matrix of movie features
    Theta = mat4["Theta"]  # 943 X 10 matrix, num_users X num_features matrix of user features
    # Compute average rating
    print("Average rating for movie 1 (Toy Story):", np.sum(Y[0, :] * R[0, :]) / np.sum(R[0, :]), "/5")
    # Reduce the data set size to run faster
    num_users, num_movies, num_features = 4, 5, 3
    X_test = X[:num_movies, :num_features]
    Theta_test = Theta[:num_users, :num_features]
    Y_test = Y[:num_movies, :num_users]
    R_test = R[:num_movies, :num_users]
    params = np.append(X_test.flatten(), Theta_test.flatten())
    # Evaluate cost function
    J, grad = cofi_cost_function(params, Y_test, R_test, num_users, num_movies, num_features, 0)
    print("Cost at loaded parameters:", J)
    J2, grad2 = cofi_cost_function(params, Y_test, R_test, num_users, num_movies, num_features, 1.5)
    print("Cost at loaded parameters (lambda = 1.5):", J2)
    # load movie list

    data_path = FileUtils.get_abs_path(__file__, "./data/movie_ids.txt")

    movieList = open(data_path, "r").read().split("\n")[:-1]
    # see movie list

    # Initialize my ratings
    my_ratings = np.zeros((1682, 1))
    # Create own ratings
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[82] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    print("New user ratings:\n")
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print("Rated", int(my_ratings[i]), "for index", movieList[i])

    Y = np.hstack((my_ratings, Y))
    R = np.hstack((my_ratings != 0, R))
    # Normalize Ratings
    Ynorm, Ymean = normalize_ratings(Y, R)

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10
    # Set initial Parameters (Theta,X)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)
    initial_parameters = np.append(X.flatten(), Theta.flatten())
    Lambda = 10

    options = {'maxiter': 100}
    result = op.minimize(fun=cofi_cost_function, x0=initial_parameters,
                         args=(Ynorm, R, num_users, num_movies, num_features, Lambda), method='TNC',
                         jac=True, options=options)
    paramsFinal = result.x

    X = paramsFinal[0:num_movies * num_features].reshape(num_movies, num_features)
    Theta = paramsFinal[num_movies * num_features:].reshape(num_users, num_features)

    p = X @ Theta.T
    my_predictions = p[:, 0][:, np.newaxis] + Ymean

    df = pd.DataFrame(np.hstack((my_predictions, np.array(movieList)[:, np.newaxis])))
    df.sort_values(by=[0], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Top recommendations for you:\n")
    for i in range(10):
        print("Predicting rating", round(float(df[0][i]), 1), " for index", df[1][i])
