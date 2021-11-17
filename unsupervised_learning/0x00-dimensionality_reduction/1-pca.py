#!/usr/bin/env python3
"""Contains the function pca()"""

import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset

    Args:
        X (ndarray)(n,d): input data
            n: no. data points
            d: no. dimensions in each point
        ndim (int): new dimensionality of the tranformed X

    Returns:
        T (ndarray)(n,ndim): contains the transformed version of X
    """
    X_meaned = X - np.mean(X, axis=0)
    U, Sigma, Vh = np.linalg.svd(X_meaned)
    Vh = Vh.T
    W = Vh[:, :ndim]

    return np.dot(X_meaned, W)
