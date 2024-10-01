import numpy as np

# Create a sample NumPy array
np.random.seed(42)  # For reproducibility
X = np.random.rand(5, 10)  # 5 rows, 10 columns

print("Original array:")
print(X)

# Indices of the columns to shuffle
columns_to_shuffle = [2, 5, 7]

# Shuffle the selected columns
shuffled_columns = X[:, columns_to_shuffle].copy()
np.random.shuffle(shuffled_columns.T)  # Shuffle the columns in place

# Replace the original columns with the shuffled columns
X[:, columns_to_shuffle] = shuffled_columns

print("\nArray after shuffling columns 2, 5, and 7:")
print(X)


#######################
np.random.permutation(X.shape[0])

def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]
Xs=[]
Xs = (np.mean(images, axis=3)).reshape(n_samples, -1)
# Scale features
Xs -= np.mean(Xs, axis=0)
Xs /= np.std(Xs, axis=0)
Swap(Xs, 0, 1)

X[0]
Xs[0]
Xs[np.random.permutation(10)].shape

Xs = (np.mean(images, axis=3)).reshape(n_samples, -1)
# Scale features
Xs -= np.mean(Xs, axis=0)
Xs /= np.std(Xs, axis=0)
columns_to_shuffle = np.random.randint(0,380,380)
columns_to_shuffle
# Shuffle the selected columns
shuffled_columns = Xs[:, columns_to_shuffle]
np.random.shuffle(shuffled_columns.T)  # Shuffle the columns in place

# Replace the original columns with the shuffled columns
Xs[:, columns_to_shuffle] = shuffled_columns
Xs
X
sum(sum(X != Xs) != 0)
Xs[:,np.random.permutation(Xs.shape[1])]
run_svm_cv(Xs[:,np.random.permutation(Xs.shape[1])], y, random_seed=42)
###################


Xs = (np.mean(images, axis=3)).reshape(n_samples, -1)
# Scale features
Xs -= np.mean(Xs, axis=0)
Xs /= np.std(Xs, axis=0)
sigma = 1

columns_to_shuffle = np.random.randint(0,380,380)
shuffled_columns = X_noisy[:, columns_to_shuffle]
np.random.shuffle(shuffled_columns.T)  # Shuffle the columns in place

# Replace the original columns with the shuffled columns
X_noisy[:, columns_to_shuffle] = shuffled_columns

run_svm_cv(X_noisy, y, random_seed=42)


n_components = 20  # jouer avec ce parametre
pca = PCA(n_components=n_components).fit(X)
X_pca = pca.transform(X_noisy)
run_svm_cv(X_pca, y)
