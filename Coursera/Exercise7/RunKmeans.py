from Coursera.Exercise7.ComputeCentroids import compute_centroids
from Coursera.Exercise7.FindClosestCentroids import find_closest_centroids


def run_kmeans(X, initial_centroids, num_iters, K):
    centroids = []
    idx = find_closest_centroids(X, initial_centroids)

    for i in range(num_iters):
        centroids = compute_centroids(X, idx, K)

        idx = find_closest_centroids(X, initial_centroids)

    return centroids, idx
