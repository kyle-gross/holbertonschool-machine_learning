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

    def pdf(self, x):
        """Calculates the PDF at a data point

        Args:
            x (ndarray)(d,1): contains data point to calculate PDF from
                d: no. dimensions of the Multnomial instance

        Returns:
            pdf: value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        multiplier = np.matmul(
            np.matmul((x - self.mean).T, inv),
            (x - self.mean)
        )
        pdf *= np.exp(-0.5 * multiplier)

        return pdf[0][0]
