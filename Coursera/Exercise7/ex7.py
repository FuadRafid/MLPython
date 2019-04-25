import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from Coursera.Exercise7.ComputeCentroids import compute_centroids
from Coursera.Exercise7.FindClosestCentroids import find_closest_centroids
from Coursera.Exercise7.InitRandomCentroid import init_random_centroid
from Coursera.Exercise7.PlotKmeans import plot_kmeans
from Coursera.Exercise7.RunKmeans import run_kmeans

mat = loadmat("data/ex7data2.mat")
X = mat["X"]
K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = find_closest_centroids(X, initial_centroids)
print("Closest centroids for the first 3 examples:\n",idx[0:3])

centroids = compute_centroids(X, idx, K)
print("Centroids computed after initial finding of closest centroids:\n", centroids)
m,n = X.shape[0],X.shape[1]
initial_centroids = init_random_centroid(X,K)
idx = find_closest_centroids(X, initial_centroids)
plot_kmeans(X, initial_centroids,idx, K,10)
plt.show()

A = plt.imread('data/bird_small.png')
A /= 255
img_size1,img_size2,rgb = A.shape
X2 = A.reshape(img_size1*img_size2, 3)

K2 = 16
num_iters = 10
initial_centroids2 = init_random_centroid(X2, K2)
centroids2, idx2 = run_kmeans(X2, initial_centroids2, num_iters,K2)

X2_recovered = centroids2[idx2, :].reshape(A.shape)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(A*255)
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X2_recovered*255)
ax[1].set_title('Compressed, with %d colors' % K2)
ax[1].grid(False)

plt.show()