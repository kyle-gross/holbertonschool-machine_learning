#!/usr/bin/env python3
"""Contains the function initialize()"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Args:
        X (ndarray)(n,d): contains dataset
            n: no. data points
            d: no. dimensions for each data point
        k (int): positive integer, contains no. clusters

    Returns:
        cetnroids (ndarray)(k,d): contains initialized centroids for each
            cluster, or None if failure
            k: no. clusters
            d: no. dimensions for each cluster
    """
    if (type(X) is not np.ndarray or type(k) is not int):
        return None

    n, d = X.shape

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))

    return centroids
