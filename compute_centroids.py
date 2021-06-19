import numpy as np

def compute_centroids(X, cluster_ids):
    """
    Function to compute the new centroid matrix
    """
    subX = []
    for x in range(len(np.unique(cluster_ids))):
        subX.append(np.array([X[i] for i in range(X.shape[0]) if cluster_ids[i] == x]))
    return np.array([np.mean(thisX,axis=0) for thisX in subX])