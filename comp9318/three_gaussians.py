from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np

n_samples = 300
centers = 3
cluster_std = 0.9
n_features = 2
random_state = 0

X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    cluster_std=cluster_std,
    n_features=n_features,
    random_state=random_state,
)
colors = ("blue", "red", "black", "green")

for k in range(centers):
    inds = np.argwhere(y == k).flatten()
    plt.scatter(X[inds, 0], X[inds, 1], color=colors[k])

plt.show()

data = np.concatenate((X, np.expand_dims(y, 1)), axis=1)
np.savetxt(
    "/Users/mkhalilia/src/github/birzeit/comp9318/data/3gaussians-std0.9.csv",
    data,
    delimiter=",",
)
