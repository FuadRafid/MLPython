import numpy as np
def poly_features(x, p , plotfit = False):
    m=np.size(x,0)
    if plotfit:
        x=x.reshape(m,1)
    x_poly = np.array([]).reshape(m,0)
    for i in range(1, 1 + p):
        x_poly=np.hstack([x_poly,np.power(x,i)])
    return x_poly
