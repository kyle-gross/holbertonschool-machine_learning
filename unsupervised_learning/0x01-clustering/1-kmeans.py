#!/usr/bin/env python3
"""Contains the function kmeans()"""

from typing import DefaultDict
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset

    Args:
        X (ndarray)(n,d): contains dataset
            n: no. data points
            d: no. dimensions for each data point
        k (int): positive int containing no. clusters
        iterations (int): number of iterations that should be performed

    Returns:
        C, clss
            C (ndarray)(k,d): contains the centroid means for each cluster
            clss (ndarray)(n,): contains index of the cluster C that each data
                point belongs to
        None, None if failure
    """
    if (type(X) is not np.ndarray or type(k) is not int or k <= 0 or
       k > X.shape[0] or len(X.shape) != 2):
        return None, None

    n, d = X.shape

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))

    for i in range(iterations):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(distances, axis=0)
        prev = centroids.copy()
        for j in range(k):
            if len(X[j == clss]) == 0:
                centroids[j] = np.random.uniform(low=np.min(X, axis=0),
                                                 high=np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = np.mean(X[j == clss], axis=0)
        if np.array_equal(prev, centroids):
            break

    return centroids, clss
