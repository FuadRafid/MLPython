import matplotlib.pyplot as plt
from scipy.io import loadmat

from Coursera.Exercise7.FeatureNormalize import feature_normalize
from Coursera.Exercise7.PCA import pca, project_data, recover_data, project_data_optimal_K

mat3 = loadmat("data/ex7data1.mat")
X3 = mat3["X"]
plt.scatter(X3[:, 0], X3[:, 1], marker="o", facecolors="none", edgecolors="b")

X_norm, mu, std = feature_normalize(X3)
U, S = pca(X_norm)[:2]
plt.scatter(X3[:, 0], X3[:, 1], marker="o", facecolors="none", edgecolors="b")
plt.plot([mu[0], (mu + 1.5 * S[0] * U[:, 0].T)[0]], [mu[1], (mu + 1.5 * S[0] * U[:, 0].T)[1]], color="black",
         linewidth=3)
plt.plot([mu[0], (mu + 1.5 * S[1] * U[:, 1].T)[0]], [mu[1], (mu + 1.5 * S[1] * U[:, 1].T)[1]], color="black",
         linewidth=3)
plt.xlim(-1, 7)
plt.ylim(2, 8)
plt.show()

print("Top eigenvector U(:,1) =:", U[:, 0])

K = 1
Z = project_data(X_norm, U, K)
print("Projection of the first example:", Z[0][0])

X_rec = recover_data(Z, U, K)
print("Approximation of the first example:", X_rec[0, :])

plt.scatter(X_norm[:, 0], X_norm[:, 1], marker="o", label="Original", facecolors="none", edgecolors="b", s=15)
plt.scatter(X_rec[:, 0], X_rec[:, 1], marker="o", label="Approximation", facecolors="none", edgecolors="r", s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()
plt.show()

mat4 = loadmat("data/ex7faces.mat")
X4 = mat4["X"]
m, n = X4.shape
print(n)
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(30, 30))
for i in range(0, 100, 10):
    for j in range(10):
        ax[int(i / 10), j].imshow(X4[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        ax[int(i / 10), j].axis("off")
plt.show()

X_norm2 = feature_normalize(X4)[0]
# Run PCA
U2, S = pca(X_norm2)

U_reduced = U2[:, :36].T
fig2, ax2 = plt.subplots(6, 6, figsize=(12, 12))
for i in range(0, 36, 6):
    for j in range(6):
        ax2[int(i / 6), j].imshow(U_reduced[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        ax2[int(i / 6), j].axis("off")
plt.show()

Z2, K2 = project_data_optimal_K(X_norm2,U2, S)
print("The projected data Z has a size of:", Z2.shape)
X_rec2 = recover_data(Z2, U2, K2)
fig3, ax3 = plt.subplots(10, 10, figsize=(20, 20))
for i in range(0, 100, 10):
    for j in range(10):
        ax3[int(i / 10), j].imshow(X_rec2[i + j, :].reshape(32, 32, order="F"), cmap="gray")
        ax3[int(i / 10), j].axis("off")
plt.show()
