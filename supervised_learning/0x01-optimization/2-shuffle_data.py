#!/usr/bin/env python3
"""Contains the function shuffle_data
"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices

    Args:
        X: numpy.ndarray - shape: (m, nx)
          * m = number of data points
          * nx = number of features in X
        Y: numpy.ndarray - shape: (m, ny)

    Returns:
        shuffled X and Y matrices
    """
    shuffle = np.random.permutation(X.shape[0])

    return X[shuffle], Y[shuffle]
