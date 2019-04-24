import matplotlib.pyplot as plt
import numpy as np


def plot_svc(svc, X):
    X_1, X_2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 1].max(), num=100),
                           np.linspace(X[:, 1].min(), X[:, 1].max(), num=100))
    plt.contour(X_1, X_2, svc.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape), 1, colors="b")
    plt.xlim(X_1.min(), X_1.max())
    plt.ylim(X_2.min(), X_2.max())
    plt.show()
