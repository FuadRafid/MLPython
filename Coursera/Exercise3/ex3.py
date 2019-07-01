import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import MLBasics as ML
from Coursera.Exercise3.LrCostFunction import cost_function_regularized
from Coursera.Exercise3.OneVsAll import one_vs_all
from Coursera.Exercise3.PredictOneVsAll import predict_one_vs_all
from utils.file_utils import FileUtils



def run():
    np.set_printoptions(suppress=True)

    data_path = FileUtils.get_abs_path(__file__, "./data/ex3data1.mat")
    mat = loadmat(data_path)
    X = mat["X"]
    y = mat["y"]
    fig, axis = plt.subplots(10, 10, figsize=(12, 12))
    for i in range(10):
        for j in range(10):
            axis[i, j].imshow(X[np.random.randint(0, 5001), :].reshape(20, 20, order="F"),
                              cmap="hot")  # reshape back to 20 pixel by 20 pixel
            axis[i, j].axis("off")
    plt.show()

    theta_t = ML.str2arr('[-2; -1; 1; 2]')
    X_t = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3, 5).T
    X_t = np.hstack((np.ones((5, 1)), X_t))
    y_t = (ML.str2arr('[1;0;1;0;1]'))
    lambda_t = 3
    cost, grad = cost_function_regularized(theta_t, X_t, y_t, lambda_t)

    print("Cost:", cost, "Expected cost: 2.534819")
    print("Gradients:\n", grad, "\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")

    lambda_value = 0.1
    num_labels = 10
    all_theta = one_vs_all(X, y, num_labels, lambda_value)
    res = predict_one_vs_all(all_theta, X)
    print("Accuracy on training set with OneVsAll:", np.mean((res == y)) * 100)


