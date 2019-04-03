import numpy as np
def sigmoid(z):
    denominator= 1 + np.exp(-z)
    ans= 1 / denominator
    return ans