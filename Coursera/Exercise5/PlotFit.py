import numpy as np
import matplotlib.pyplot as plt
from Coursera.Exercise5.PolyFeatures import poly_features
def plot_fit(min_x, max_x, mu, sigma, theta, p):

    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05))
    # Map the X values
    X_poly = poly_features(x, p , True)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma

    # Add ones
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)
