#!/usr/bin/env python3
"""Contains the function batch_norm()
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using
    batch normalization

    Args:
        Z: numpy.ndarray, shape: (m, n) - array to normalize
            * m: number of data points
            * n: number of features in Z
        gamma: numpy.ndarray, shape (1, n) - contains scales used for
            batch normalization
        beta: numpy.ndarray, shape (1, n) - contains the offsets used
            for batch normalization
        epsilon: small number to avoid division by 0

    Returns:
        normalized Z matrix
    """
    variance, mean = np.var(Z, axis=0), np.mean(Z, axis=0)
    Z_norm = (Z - mean) / (np.sqrt(variance + epsilon))

    return gamma * Z_norm + beta
