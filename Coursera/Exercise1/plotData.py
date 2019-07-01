import matplotlib.pyplot as plt
import numpy as np

from utils.file_utils import FileUtils


def plot_data():
    data_path = FileUtils.get_abs_path(__file__, "./data/ex1data1.txt")
    data = np.loadtxt(data_path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, marker='x', cmap='red')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel('Profit in $10,000s')
