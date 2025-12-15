import numpy as np
from sklearn.decomposition import PCA

def pca(X_train: int, X_test: int, randomized: bool, seed: int, target: float):
    """
    Docstring for pca
    
    :param X_train: input vector of train features
    :param X_test: input vector of test features
    :bool randomized: Wheter the data is shuffled or not
    :int seed: Fixed seed to make experiments repetable
    :float target: Target variance to be explained
    """
    
    # initailaze PCA
    reducer = PCA(
        n_components=X_train.shape[1]-1,
        svd_solver="randomized",
        iterated_power=7 if randomized else "auto",
        random_state=seed
    )
    
    # fit PCA on training data
    reducer.fit(X_train)
    evr = reducer.explained_variance_ratio_
    cum = np.cumsum(evr)

    # find number of components to reach target variance
    k = int(np.searchsorted(cum, target) + 1)

    reducer = PCA(n_components=k, svd_solver="randomized", iterated_power=7, random_state=42)

    # transform data
    Z_train = reducer.fit_transform(X_train)
    Z_test = reducer.transform(X_test)
    
    return Z_train, Z_test