import numpy as np
from find_closest_centroids import find_closest_centroids
from compute_centroids import compute_centroids

def k_means(X, initial_centroids, K, max_iter):
    """
    Function performing the algorithm
    """
    centroid_history = []
    current_centroids = initial_centroids
    for i in range(max_iter):
        print("Running iteration number:", i+1)
        centroid_history.append(current_centroids)
        cluster_ids = find_closest_centroids(X, current_centroids)
        current_centroids = compute_centroids(X, cluster_ids)
    return cluster_ids, centroid_history