import numpy as np
from scipy.io import loadmat

from Coursera.Exercise3.NNPrediction import predict_nn

mat2 = loadmat("data/ex3weights.mat")
Theta1=mat2['Theta1']
Theta2=mat2['Theta2']

np.set_printoptions(suppress=True)
mat = loadmat("data/ex3data1.mat")
X = mat["X"]
y = mat["y"]

res=predict_nn(Theta1,Theta2,X)
print("Accuracy on training set with Neural Network:", np.mean((res == y)) * 100)