from sklearn.datasets import make_circles, make_moons
from matplotlib import pyplot as plt
import numpy as np

colors = ("blue", "red", "black", "green")

####################
# Cricles
####################
X1, y1 = make_circles(factor=0.5, noise=0.05, n_samples=1000)

for k in range(2):
    inds = np.argwhere(y1 == k).flatten()
    plt.scatter(X1[inds, 0], X1[inds, 1], color=colors[k])

plt.show()

data = np.concatenate((X1, np.expand_dims(y1, 1)), axis=1)
np.savetxt(
    "/Users/mkhalilia/src/github/birzeit/comp9318/data/circles.csv",
    data,
    delimiter=",",
)

####################
# Moons
####################
X2, y2 = make_moons(n_samples=1000, noise=0.05)

for k in range(2):
    inds = np.argwhere(y2 == k).flatten()
    plt.scatter(X2[inds, 0], X2[inds, 1], color=colors[k])

plt.show()

data = np.concatenate((X2, np.expand_dims(y2, 1)), axis=1)
np.savetxt(
    "/Users/mkhalilia/src/github/birzeit/comp9318/data/moons.csv",
    data,
    delimiter=",",
)