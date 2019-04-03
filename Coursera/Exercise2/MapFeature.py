import numpy as np


def map_features(X1, X2):
    degree = 6
    n=len(X1)
    X1=X1.reshape([n,1])
    n = len(X2)
    X2 = X2.reshape([n, 1])
    out = np.ones(np.shape(X1))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out=np.hstack([out, (X1 ** (i - j)) * (X2 ** j)])
    return out

