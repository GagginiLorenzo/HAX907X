n1=100
n2=100
sigma=0.1
sigma2=2
import numpy as np
from functools import partial  # useful for weighted distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
from scipy import stats  # to use scipy.stats.mode
from sklearn import neighbors
from sklearn import datasets

nbp = int(np.floor(n1 / 8))
nbp
nbn = int(np.floor(n2 / 8))
nbn
xapp = np.reshape(np.random.rand((nbp + nbn) * 16), [(nbp + nbn) * 8, 2])
xapp
np.random.rand((nbp + nbn) * 16)
[(nbp + nbn) * 8, 2]
yapp = np.ones((nbp + nbn) * 8)
idx = 0
for i in range(-2, 2):
    for j in range(-2, 2):
        if (((i + j) % 2) == 0):
            nb = nbp
        else:
            nb = nbn
            yapp[idx:(idx + nb)] = [(i + j) % 3 + 1] * nb

        xapp[idx:(idx + nb), 0] = np.random.rand(nb)
        xapp[idx:(idx + nb), 0] += i + sigma * np.random.randn(nb)
        xapp[idx:(idx + nb), 1] = np.random.rand(nb)
        xapp[idx:(idx + nb), 1] += j + sigma * np.random.randn(nb)
        idx += nbp

ind = np.arange((nbp + nbn) * 8)
np.random.shuffle(ind)
res = np.hstack([xapp, yapp[:, np.newaxis]])
#np.array(res[ind, :2]), np.array(res[ind, 2])

m=np.array([1,2,3,4,5,6])
n=m.reshape(3,2)

dist = metrics.pairwise.pairwise_distances(n)
dist
n
idx_sort = np.argsort(dist)
idx_sort
class KNNClassifier(BaseEstimator, ClassifierMixin):
    """Home made KNN Classifier class."""

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        n_samples, n_features = X.shape
        # TODO: Compute all pairwise distances between X and self.X_ using e.g. metrics.pairwise.pairwise_distances
        dist = metrics.pairwise.pairwise_distances(X,self.X_)
        # Get indices to sort them
        idx_sort = np.argsort(dist)
        # Get indices of neighbors
        idx_neighbors =
        # Get labels of neighbors
        y_neighbors =
        # Find the predicted labels y for each entry in X
        # You can use the scipy.stats.mode function
        mode, _ =
        # the following might be needed for dimensionaality
        y_pred = np.asarray(mode.ravel(), dtype=np.intp)
        return y_pred
