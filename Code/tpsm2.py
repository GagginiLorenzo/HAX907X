import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score,cross_validate
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
#plt.savefig("visage.png")
plt.show()

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

Cs =  np.geomspace(0.00001,100000,30)
scores=[]
for C in Cs :
    clf=SVC(kernel='linear',C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_train, y_train))
scores

ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
#plt.savefig("regularization_scores.png")
plt.show()

print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()

clf=SVC(kernel='linear',C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))



#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

####################################################################
####################################################################

#Q5

def run_svm_cv(_X, _y, random_seed=None):
    if random_seed is not None:
            np.random.seed(random_seed)
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    # Extract training and test scores
    print("Cross validate Train scores: ", _clf_linear.score(_X_train, _y_train))
    print("Cross validate Test scores: ", _clf_linear.score(_X_test, _y_test),"\n")

sigma = 1 #On laisse sigma à 1 d'abord, important car les données sont centrées et réduites, se serra notre référence.
seed=1

print("Score sans variable de nuisance\n")
run_svm_cv(X, y, random_seed=seed)

for i in range(1, 10,1):
    np.random.seed(seed)
    X_noisy = []
    noise = sigma * np.random.randn(n_samples, i*10000)
    X_noisy = np.concatenate((X, noise), axis=1)

    print("--------------------------------------------------")
    print("Score avec ",i*10000," variable de nuisance de variance :",sigma,", signal dilué a :",1/(1+i)," \n")
    run_svm_cv(X_noisy, y, random_seed=seed)


for i in range(1, 5,1):
    sigma = i
    np.random.seed(seed)
    X_noisy = []
    noise = sigma * np.random.randn(n_samples, 10000)
    X_noisy = np.concatenate((X, noise), axis=1)

    print("--------------------------------------------------")
    print("Score avec ",10000," variable de nuisance de variance ",sigma,", signal initial dilué a :",0.5," \n")
    run_svm_cv(X_noisy, y, random_seed=seed)

# on conclue que le nombre de variables de nuisance à moins d'impact sur le score de prédiction que leur variance '
# remarque : on a ajouter un bruit non centré, faire attention avec la comparaison initiale...

"""
noise = sigma * np.random.randn(n_samples, t)
X_noisy = np.concatenate((Xs, noise), axis=1)

for t in range(300,3300,300):
    X_noisy=[]
    noise = sigma * np.random.randn(n_samples, t)
    X_noisy = np.concatenate((Xs, noise), axis=1)
    print(X_noisy.shape)
    run_svm_cv(X_noisy, y, random_seed=42)
    print(t)
"""

noise1 = sigma * np.random.randn(n_samples, 10000//2)
noise2 = noise1 * np.random.randn(n_samples, 10000//2)
noise=np.concatenate((noise1, noise2), axis=1)
noise.shape

X_noisy = []
noise = sigma * np.random.randn(n_samples, 40000)
X_noisy = np.concatenate((X, noise), axis=1)
n_components = 20  # jouer avec ce parametre
pca = PCA(n_components=n_components).fit(X_noisy)
X_pca = pca.transform(X_noisy)
run_svm_cv(X_pca, y)
run_svm_cv(X_noisy, y)
