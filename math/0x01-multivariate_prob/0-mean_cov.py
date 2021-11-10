#!/usr/bin/env python3
"""Contains the function mean_cov()"""

import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set

    Args:
        X (ndarray)(n,d): contains data set
            n: number of data points
            d: number of dimensions in each data point

    Returns:
        mean (ndarray)(1,d): mean of the data set
        cov (ndarray)(d,d): covariance of the data set
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n, d = X.shape
    
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0, keepdims=True)
    cov = 1 / (n - 1) * np.matmul(X.T - mean.T, X - mean)

    return mean, cov
