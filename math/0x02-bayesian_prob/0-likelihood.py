#!/usr/bin/env python3
"""Contains the function likelihood()"""

import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of patients developing severe side effects

    Args:
        x (int): no. patients that develop severe side effects
        n (int): no. patients observed
        P (1D ndarray): contains various hypothetical probabilities of
            developing severe side effects

    Returns:
        1D numpy.ndarray: likelihood of obtaining the data, x, and n for
            each probability in P
    """
    if n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not (np.all(P >= 0) and np.all(P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

    factorial = np.math.factorial

    likelihood = (
        factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
        ((1 - P) ** (n - x))
    )

    return likelihood
