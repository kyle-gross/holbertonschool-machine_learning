#!/usr/bin/env python3
"""Contains the function correlation()"""

import numpy as np
from numpy.lib.twodim_base import diag


def correlation(C):
    """Calculates a correlation matrix

    Args:
        C (ndarray)(d,d): contains a covariance matrix
            d: number of dimensions

    Returns:
        corr (ndarray)(d,d): contains correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    diag = np.sqrt(np.diag(C))
    diag_inv = 1 / np.outer(diag, diag)
    corr = diag_inv * C

    return corr
