#!/usr/bin/env python3
"""Contains the function definiteness()"""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix

    Args:
        matrix (ndarray)(n,n): calculate the definiteness of

    Returns:
        string representing definiteness of the matrix
    """
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigvals = np.linalg.eigvals(matrix)

    if np.all(eigvals > 0):
        return "Positive definite"
    elif np.all(eigvals >= 0):
        return "Positive semi-definite"
    elif np.all(eigvals < 0):
        return "Negative definite"
    elif np.all(eigvals <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
