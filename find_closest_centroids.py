import numpy as np
from dist_squared import dist_squared

def find_closest_centroids(X, centroids):
    """
    Function to return the array of the closest centroid to each training example
    """
    cluster_ids = np.zeros((X.shape[0],1))
    
    for x in range(cluster_ids.shape[0]):
        point = X[x]
        min_dist, id = 9999999, 0
        for i in range(centroids.shape[0]):
            centroid = centroids[i]
            square_distance = dist_squared(centroid,point)
            if square_distance < min_dist:
                min_dist = square_distance
                id = i
        cluster_ids[x] = id
        
    return cluster_ids