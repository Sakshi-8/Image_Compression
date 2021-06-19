import numpy as np

def dist_squared(point1, point2):
    """
    Function to return the squared distance between two points
    """
    # assert point1.shape == point2.shape
    return np.sum(np.square(point2-point1))