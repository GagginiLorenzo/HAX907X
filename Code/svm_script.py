#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')


###############################################################################
#               Iris Dataset
###############################################################################

"""
Code utilisé pour la partie 1. Introduction aux SVM de Sklearn
"""

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

Cs = list(np.logspace(-3, 3, 5)) # 5 valeurs de C à tester suffisent, j'ai tester avec plus mais on est déja tres proche de la solution optimale
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]
parameters_polynomial = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_p_grid=GridSearchCV(SVC(), parameters_polynomial,n_jobs=-1)
clf_p_grid.fit(X_train, y_train)
clf_p=clf_p_grid.best_estimator_


n=1 # nombre de répétitions pour faire la moyenne (j'ai laiser 1 pour le plot, mais les résultats du rapport sont pour n=10), 
# dans les deux cas c'est le dernier estimateur qui sera plot)

# on aurait pu utiliser une cross validation mais je voulais avoir les paramètres optimaux pour chaque répétition
# et j'ai pas eu de bonne idée pour proprement.

average_performances_l = np.empty(shape=(n,2))
average_performances_p = np.empty(shape=(n,2))
param_l = list()
param_p = list()
for k in range(0,n):
    print(k)
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf_l_grid.fit(X_train, y_train)
    clf_l=clf_l_grid.best_estimator_

    clf_p_grid.fit(X_train, y_train)
    clf_p=clf_p_grid.best_estimator_
    param_l.append(clf_l_grid.best_params_)
    param_p.append(clf_p_grid.best_params_)
    average_performances_l[k] = [clf_l.score(X_train, y_train),clf_l.score(X_test, y_test)]
    average_performances_p[k] = [clf_p.score(X_train, y_train),clf_p.score(X_test, y_test)]


print("polynomial :", "train :",average_performances_p.mean(axis=0)[0] ,"test :",average_performances_p.mean(axis=0)[1])
print("linear :", "train :",average_performances_l.mean(axis=0)[0],"test :",average_performances_l.mean(axis=0)[1])


#Detailles des résultats
param_l
average_performances_l
average_performances_p
param_p


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


###############################################################################
#               Face Recognition Task
###############################################################################
"""
Code utilisé pour la partie 3. Reconnaissance de visages
"""

""""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################

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
    print("Train scores: ", _clf_linear.score(_X_train, _y_train))
    print("Test scores: ", _clf_linear.score(_X_test, _y_test),"\n")

sigma = 1 #On laisse sigma à 1 d'abord, important car les données sont centrées et réduites, se serra notre référence.
seed=1

print("Score sans variable de nuisance\n")
run_svm_cv(X, y, random_seed=seed)
#On ajoute des variables de nuisances réduites
for i in range(1, 5,1):
    np.random.seed(seed)
    X_noisy = []
    noise = sigma * np.random.randn(n_samples, i*10000)
    X_noisy = np.concatenate((X, noise), axis=1)

    print("--------------------------------------------------")
    print("Score avec ",i*10000," variable de nuisance de variance :",sigma,", signal dilué a :",1/(1+i)," \n")
    run_svm_cv(X_noisy, y, random_seed=seed)

#on ajoute des variables de nuisances non réduite
print("Score sans variable de nuisance\n")
run_svm_cv(X, y, random_seed=seed)
for i in range(1, 5,1):
    sigma = i
    np.random.seed(seed)
    X_noisy = []
    noise = sigma * np.random.randn(n_samples, 10000)
    X_noisy = np.concatenate((X, noise), axis=1)

    print("--------------------------------------------------")
    print("Score avec ",10000," variable de nuisance de variance ",sigma,", signal initial dilué a :",0.5," \n")
    run_svm_cv(X_noisy, y, random_seed=seed)

# On conclue que le nombre de variables de nuisance à moins d'impact sur le score de prédiction que leur variance '
# remarque : on a ajouter un bruit non centré, faire attention avec la comparaison précédente...
# Mais on peut imaginé ce genre de bruit dans des cas réels, par exemple des capteurs abimés qui renvoient des valeurs aberrantes.
# Même si dans se cas, on pourais facillement les repérer et les retirer... a réfléchir.

####################################################################
#Ne faite pas tourner cette cellule, elle prends beaucoup de temps.
#Test avec le solver auto et randomized
#randomized n'améliore pas les performances, mais dans mon cas, j'ai choisie de noyer le signal dans un bruit
#donc ce n'est pas le solveur adapté !

X_noisy = []
noise = sigma * np.random.randn(n_samples, 40000)
X_noisy = np.concatenate((X, noise), axis=1)
n_components = 20
pca = PCA(n_components=n_components).fit(X_noisy,solver='randomized')
X_pca = pca.transform(X_noisy)
print("score sans PCA : ")
run_svm_cv(X_noisy, y)
print("score avec PCA : ")
run_svm_cv(X_pca, y)
