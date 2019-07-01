from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from Coursera.Exercise5.FeatureNormalize import feature_normalize
from Coursera.Exercise5.LeanringCurve import learning_curve
from Coursera.Exercise5.LinearCostFunc import linear_reg_cost
from Coursera.Exercise5.PlotFit import plot_fit
from Coursera.Exercise5.PolyFeatures import poly_features
from Coursera.Exercise5.TrainLinearReg import train_linear_reg
from Coursera.Exercise5.ValidationCurve import validation_curve
from utils.file_utils import FileUtils

data_path = FileUtils.get_abs_path(__file__, "./data/ex5data1.mat")
mat = loadmat(data_path)
X = mat["X"]
y = mat["y"]
Xval=mat["Xval"]
yval=mat["yval"]
Xtest=mat["Xtest"]
ytest=mat["ytest"]
m, n = X.shape
plt.scatter(X, y)
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of dam (y)")
plt.show()

ones = np.ones([m, 1])
x = np.hstack([ones, X])

onesVal = np.ones([np.size(Xval,0), 1])
Xval_ones= np.hstack([onesVal, Xval])

theta = np.array([[1], [1]])
J, grad = linear_reg_cost(theta, x , y , 1)
print('Cost at theta = [1 ; 1]: %', J,
      '\n(this value should be about 303.993192)\n')

print('Gradient at theta = [1 ; 1]: ', grad,
      '\n(this value should be about [-15.303016; 598.250744])\n')

lambda_value = 0
theta = train_linear_reg(x, y, lambda_value)
plt.scatter(X, y)
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of dam (y)")
plt.plot(x[:, 1], x @ theta, '-', color='red')
plt.show()

lambda_value = 0
error_train,error_val=learning_curve(x, y, Xval_ones, yval, lambda_value)
plt.plot(error_val, '-', color='red')
plt.plot(error_train, '-', color='blue')
plt.title("Leaning Curve")
plt.legend(['error_val','error_train'])
plt.ylabel("Error")
plt.xlabel("No of samples")
plt.show()

p=8
x_poly = poly_features(X,p)
x_poly,mu,sigma = feature_normalize(x_poly)
ones = np.ones([m, 1])
x_poly = np.hstack([ones, x_poly])


X_poly_test = poly_features(Xtest, p)
X_poly_test=(X_poly_test-mu)/sigma


X_poly_val = poly_features(Xval, p)
X_poly_val=(X_poly_val-mu)/sigma
m_poly_val = np.size(X_poly_val,0)
ones = np.ones([m_poly_val, 1])
X_poly_val = np.hstack([ones, X_poly_val])



print('Normalized Training Example 1:\n');
print(x_poly[0, :])

lambda_value = 0
theta = train_linear_reg(x_poly, y, lambda_value)
plt.scatter(X, y,color='red')
# plt.plot(x[:, 1], x_poly @ theta, '-', color='red')

plot_fit(min(x[:, 1]),max(x[:, 1]),mu,sigma,theta,p)
plt.title("Polynomial Features Fitting")
plt.show()

error_train,error_val=learning_curve(x_poly, y, X_poly_val, yval, lambda_value)

plt.plot(range(1,m+1),error_val, '-', color='red')
plt.plot(range(1,m+1),error_train, '-', color='blue')
plt.title("Learning Curve for Polynomial Features")
plt.legend(['error_val','error_train'])
plt.ylabel("Error")
plt.xlabel("No of samples")
plt.show()

lambda_vec,error_train,error_val=validation_curve(x_poly, y, X_poly_val, yval)

plt.plot(lambda_vec,error_val, '-', color='red')
plt.plot(lambda_vec,error_train, '-', color='blue')
plt.title("Lambda vs Error for Polynomial Features")
plt.legend(['error_val','error_train'])
plt.ylabel("Error")
plt.xlabel("Lambda")
plt.show()