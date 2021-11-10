#!/usr/bin/env python3
"""Contains the class MultiNormal"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal Distribution"""
    def __init__(self, data):
        """Instantiate MultiNormal class

        Args:
            data (ndarray)(d,n): contains data set
                n: no. data points
                d: no. dimensions in each data point

        Attributes:
            mean (ndarray)(d,1): contains mean of data
            cov (ndarray)(d,d): contains covariance matrix of data
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        d, n = data.shape

        if n < 2:
            raise ValueError('data must contain multiple data points')

        mean = np.mean(data, axis=1, keepdims=True)
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)

        self.mean = mean
        self.cov = cov
