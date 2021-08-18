#!/usr/bin/env python3
"""Contains the function 'normalization_constants'
"""
import numpy as np


def normalization_constants(X):
    """Caclulates the normalization constants of a matrix

    Args:
        X: numpy.ndarray - shape: (m, nx)

    Returns:
        Mean and standard deviation of each feature
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
