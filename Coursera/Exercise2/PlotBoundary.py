from Coursera.Exercise2.MapFeature import map_features
from Coursera.Exercise2.Plotdata import plot_data
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(theta, x, y):
    if np.size(x, 1) <= 3:
        plot_data(x[:, 1:], y)
        plot_x = np.array([min(x[:, 1]) - 2, max(x[:, 1]) + 2])
        plot_y = (-1 / theta[2, 0]) * (theta[1, 0] * plot_x + theta[0, 0])
        plt.plot(plot_x, plot_y)
        plt.show()
    else:
        u = np.linspace(-1, 1.5, 50,endpoint=True)
        v = np.linspace(-1, 1.5, 50,endpoint=True)
        z = np.zeros([len(u), len(v)])
        for i in range(len(u)):
            for j in range(len(v) ):
                z[i, j] = map_features(np.array([u[i]]), np.array([v[j]])) @ theta

        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        np.array(z)
        z=np.transpose(z)
        plot_data(x[:, 1:], y)
        ax.contour(u, v, z,[0])
        plt.show()