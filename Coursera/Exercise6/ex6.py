import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC

from Coursera.Exercise6.Dataset3params import dataset_3_params
from Coursera.Exercise6.GaussianKernel import gaussian_kernel
from Coursera.Exercise6.PlotData import plot_data
from Coursera.Exercise6.PlotSVC import plot_svc

mat = loadmat("data/ex6data1.mat")
X = mat["X"]
y = mat["y"]

plot_data(X, y)

classifier = SVC(C=1, kernel="linear")
classifier.fit(X, np.ravel(y))

plot_svc(classifier, X)

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

print(gaussian_kernel(x1, x2, sigma))

data2 = loadmat('data/ex6data2.mat')

y2 = data2['y']
X2 = data2['X']

plot_data(X2, y2)

clf2 = SVC(kernel='rbf', gamma=30)
clf2.fit(X2, y2.ravel())
plot_svc(clf2, X2)

data3 = loadmat('data/ex6data3.mat')
X3 = data3["X"]
y3 = data3["y"]
Xval = data3["Xval"]
yval = data3["yval"]

plot_data(X3, y3)
C, gamma = dataset_3_params(X3, y3, Xval, yval)
clf3 = SVC(C=C, gamma=gamma)
clf3.fit(X3, y3.ravel())
plot_svc(clf3, X3)
