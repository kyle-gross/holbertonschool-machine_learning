#!/usr/bin/env python3
"""Contains the function initialize()"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model (GMM)

    Args:
        X (ndarray)(n,d): contains data set
        k (int): no. clusters

    Returns:
        pi, m, S // None, None, None if failure
            pi (ndarray)(k,): contains priors for each cluster, initialized
                evenly
            m (ndarray)(k,d): contains centroid means for each cluster,
                initialized with K-means
            S (ndarray)(k,d,d): contains covariance matrices for each
                cluster, initialized as identity matrices
    """
    if (type(X) is not np.ndarray or len(X.shape) != 2 or
       type(k) is not int or k <= 0):
        return None, None, None

    n, d = X.shape
    pi = np.ones(k) / k
    m, clss = kmeans(X, k)
    S = np.ndarray((k, d, d))
    S[:] = np.identity(d)

    return pi, m, S
