from tkinter.constants import ACTIVE
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from svm_source import *
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

Cs=list(np.logspace(-3, 3, 200))
parameters_linear = {'kernel': ['linear'], 'C': Cs}
clf_l_grid=GridSearchCV(SVC(), parameters_linear,n_jobs=-1)
clf_l_grid.fit(X_train, y_train)
clf_l=clf_l_grid.best_estimator_

Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]
parameters_polynomial = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_p_grid=GridSearchCV(SVC(), parameters_polynomial,n_jobs=-1)
clf_p_grid.fit(X_train, y_train)
clf_p=clf_p_grid.best_estimator_

print("polynomial :", "train :",clf_p.score(X_train, y_train),"test :",clf_p.score(X_test, y_test))
print("linear :", "train :",clf_l.score(X_train, y_train),"test :",clf_l.score(X_test, y_test))

n=1
average_performances_l = np.empty(shape=(n,2))
average_performances_p = np.empty(shape=(n,2))
for k in range(0,n):
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf_l_grid.fit(X_train, y_train)
    clf_l=clf_l_grid.best_estimator_

    clf_p_grid.fit(X_train, y_train)
    clf_p=clf_p_grid.best_estimator_

    average_performances_l[k] = [clf_l.score(X_train, y_train),clf_l.score(X_test, y_test)]
    average_performances_p[k] = [clf_p.score(X_train, y_train),clf_p.score(X_test, y_test)]

average_performances_l
average_performances_p

average_performances_l.mean(axis=0)
average_performances_p.mean(axis=0)
clf_p_grid.best_params_
clf_l_grid.best_params_

#mean_performances = np.mean(average_performances, axis=0)

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_l.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_p.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()
