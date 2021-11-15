#!/usr/bin/env python3
"""Contains the functions likelihood() and intersection()"""

import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of patients developing severe side effects

    Args:
        x (int): no. patients that develop severe side effects
        n (int): no. patients observed
        P (1D ndarray): contains various hypothetical probabilities of
            developing severe side effects

    Returns:
        likelihood (1D ndarray): likelihood of obtaining the data, x, and n
            for each probability in P
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not (np.all(P >= 0) and np.all(P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

    factorial = np.math.factorial

    likelihood = (
        factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
        ((1 - P) ** (n - x))
    )

    return likelihood


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining the data with various
    hypothetical probabilities

    Args:
        x (int): no. patients who develop severe side effects
        n (int): no. patients observed
        P (1D ndarray): contains the various hypothetical probabilities of
            developing severe side effects
        Pr (1D ndarray): contains prior beliefs of P

    Returns:
        intersection (1D ndarray): containing the intersection of obtaining x
            and n with each probability in P, respectively
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError(
            'Pr must be a  numpy.ndarray with the same shape as P'
        )
    if not (np.all(P >= 0) and np.all(P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if not (np.all(Pr >= 0) and np.all(Pr <= 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(1, np.sum(Pr)):
        raise ValueError('Pr must sum to 1')

    factorial = np.math.factorial

    likelihood = (
        factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
        ((1 - P) ** (n - x))
    )

    intersection = likelihood * Pr

    return intersection
