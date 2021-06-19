import numpy as np
from random import sample #Used for random initialization

def choose_k_random_centroids(X, K):
    """
    Function to return K random centroids from the training examples
    """
    random_indices = sample(range(0,X.shape[0]),K)
    return np.array([X[i] for i in random_indices])