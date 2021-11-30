#!/usr/bin/env python3
"""Contains the function optimum_k()"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance

    Args:
        X (ndarray)(n,d): contains data set
        kmin (int): contains minimum no. clusters to check for
        kmax (int): contains maximum no. clusters to check for
        iterations (int): contains maximum no. iterations for K-means

    Returns:
        results, dvars // None, None if failure
            results (list): contains the outputs of K-means for each cluster
                size
            d_vars (list): contains difference in variance from the smallest
                cluster for each cluster size
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if (type(kmin) is not int or kmin < 1 or
       type(kmax) is not int or kmax < 1 or
       type(iterations) is not int or iterations < 1 or
       kmin >= kmax):
        return None, None
    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, results[0][0]) - variance(X, C))

    return results, d_vars
