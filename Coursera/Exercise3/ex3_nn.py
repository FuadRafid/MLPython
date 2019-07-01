import numpy as np
from scipy.io import loadmat

from Coursera.Exercise3.NNPrediction import predict_nn
from utils.file_utils import FileUtils


def run():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex3weights.mat")
    mat2 = loadmat(data_path)
    Theta1=mat2['Theta1']
    Theta2=mat2['Theta2']

    np.set_printoptions(suppress=True)
    data_path = FileUtils.get_abs_path(__file__, "./data/ex3data1.mat")
    mat = loadmat(data_path)
    X = mat["X"]
    y = mat["y"]

    res=predict_nn(Theta1,Theta2,X)
    print("Accuracy on training set with Neural Network:", np.mean((res == y)) * 100)