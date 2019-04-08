import numpy as np
import scipy.optimize as op

from Coursera.Exercise3.LrCostFunction import cost_function_regularized


def one_vs_all(x, y, num_labels, lmbda):
    m, n = x.shape
    ones = np.ones([m, 1])
    x = np.hstack([ones, x])
    all_theta = []
    for i in range(1,num_labels+1):
        initial_theta = np.zeros([n + 1, 1])
        options={'maxiter':50}
        result = op.minimize(fun=cost_function_regularized, x0=initial_theta, args=(x, np.where(y==i,1,0), lmbda), method='TNC',
                             jac=True,options=options)
        optimal_theta = result.x
        all_theta.append(optimal_theta)
    return np.array(all_theta)
