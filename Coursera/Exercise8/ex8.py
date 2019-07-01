import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


from Coursera.Exercise8.EstimateGaussian import estimate_gaussian
from Coursera.Exercise8.MutlivariateGaussian import multivariate_gaussian
from Coursera.Exercise8.PlotData import visualize_fit
from Coursera.Exercise8.SelectThreshold import select_threshold
from utils.file_utils import FileUtils


def run():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex8data1.mat")
    mat = loadmat(data_path)
    X = mat["X"]
    Xval = mat["Xval"]
    yval = mat["yval"]

    plt.scatter(X[:, 0], X[:, 1], marker="x")
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.show()

    mu, sigma2 = estimate_gaussian(X)

    p = multivariate_gaussian(X, mu, sigma2)

    visualize_fit(X, mu, sigma2)

    pval = multivariate_gaussian(Xval, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)
    print("Best epsilon found using cross-validation:",epsilon)
    print("Best F1 on Cross Validation Set:",F1)

    outliers = np.nonzero(p<epsilon)[0]
    plt.scatter(X[outliers,0],X[outliers,1],marker ="o",facecolor="none",edgecolor="r",s=70)
    plt.xlim(0,35)
    plt.ylim(0,35)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")

    plt.show()
