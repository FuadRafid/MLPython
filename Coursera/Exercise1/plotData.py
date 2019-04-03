import matplotlib.pyplot as plt
import numpy as np


def plot_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, marker='x', cmap='red')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel('Profit in $10,000s')
