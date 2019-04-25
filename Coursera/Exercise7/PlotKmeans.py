import matplotlib.pyplot as plt

from Coursera.Exercise7.ComputeCentroids import compute_centroids
from Coursera.Exercise7.FindClosestCentroids import find_closest_centroids


def plot_kmeans(X, centroids, idx, K, num_iters):
    """
    plots the data points with colors assigned to each centroid
    """
    m, n = X.shape[0], X.shape[1]

    fig, ax = plt.subplots(nrows=num_iters, ncols=1, figsize=(6, 36))

    for i in range(num_iters):
        # Visualisation of data
        color = "rgb"
        ax[i].scatter(X[:, 0], X[:, 1])
        for k in range(1, K + 1):
            grp = (idx == k).reshape(m, 1)
            ax[i].scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k - 1], s=15)
        # visualize the new centroids
        ax[i].scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        title = "Iteration Number " + str(i)
        ax[i].set_title(title)

        # Compute the centroids mean
        centroids = compute_centroids(X, idx, K)

        # assign each training example to the nearest centroid
        idx = find_closest_centroids(X, centroids)

    plt.tight_layout()
