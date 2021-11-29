#!/usr/bin/env python3
"""Contains the function variance()"""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set

    Args:
        X (ndarray)(n,d): contains data set
        C (ndarray)(k,d): contains centroid means for each cluster

    Returns:
        var (float): total variance
        None if falure
    """
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(C) is not np.ndarray or C.ndim != 2):
        return None

    distances = np.linalg.norm(X - np.expand_dims(C, 1), axis=-1)
    min = distances.min(axis=0)
    var = np.sum(np.square(min))

    return var
