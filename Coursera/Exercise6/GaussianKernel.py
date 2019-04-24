import numpy as np


def gaussian_kernel(x1, x2, sigma=2):
    e = np.sum(np.power(x1 - x2, 2))
    e = e / (2 * (sigma ** 2))
    sim = np.exp(-e)
    return sim
