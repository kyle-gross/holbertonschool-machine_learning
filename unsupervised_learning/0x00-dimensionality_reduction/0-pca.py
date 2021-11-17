#!/usr/bin/env python3
"""Contains the function pca()"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset

    Args:
        X (ndarray)(n,d): contains data
            n: no. data points
            d: no. dimensions in each point
        var (float): fraction of the variance that the PCA transformation
            should maintain

    Returns:
        W (ndarray)(d,nd): weights matrix that maintains var fraction of X's
            original variance
            d: no. dimensions in each point
            nd: new dimensionality of the transformed X
    """
    U, Sigma, Vh = np.linalg.svd(X)
    variance_cumsum = np.cumsum(Sigma) / np.sum(Sigma)
    nd = np.argwhere(variance_cumsum >= var)
    W = Vh[:(nd[0, 0]+1)].T

    return W
