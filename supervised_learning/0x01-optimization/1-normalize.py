#!/usr/bin/env python3
"""Contains the function 'normalize'
"""
import numpy as np


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix

    Args:
        X: numpy.ndarray - shape: (d, nx)
            d: number of data points
            nx: number of features
        m: numpy.ndarray - shape: (nx,) - contains the mean of
           all features of X
        s: numpy.ndarray - shape: (nx,) - contains the std dev of
           all features of X

    Returns:
        Normalized X matrix
    """
    return (X - m) / s
