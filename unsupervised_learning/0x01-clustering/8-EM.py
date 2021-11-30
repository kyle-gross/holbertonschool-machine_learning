#!/usr/bin/env python3
"""Contains the function expectation_maximization()"""

import numpy as np
expectation = __import__('6-expectation').expectation
initialize = __import__('4-initialize').initialize
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM

    Args:
        X (ndarray)(n,d): contains data set
        k (int): contains no. clusters
        iterations (int): contains max no. iterations for the algorithm
        tol (float): tolerance of the log likelihood. Used to determine
            early stopping
        verbose (bool): if True, prints information about the algorithm

    Returns:
        pi, m, S, g, L // None, None, None, None, None if failure
        pi (ndarray)(k,): priors for each cluster
        m (ndarray)(k,d): centroid means for each cluster
        S (ndarray)(k,d,d): covariance matrices for each cluster
        g (ndarray)(k,n): probabilities for each data point in each cluster
        l (flaot): log likelihood of the model
    """
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(k) is not int or k <= 0 or k > X.shape[0] or
       type(iterations) is not int or iterations <= 0 or
       type(tol) is not float or tol < 0 or
       type(verbose) is not bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    logL_prev = 0

    for i in range(iterations):
        g, logL = expectation(X, pi, m, S)
        if g is None or logL is None or pi is None or m is None or S is None:
            return None, None, None, None, None
        if (verbose and ((i % 10 == 0 or i == iterations) or
           abs(logL - logL_prev) <= tol)):
            print('Log Likelihood after {} iterations: {}'.format(
                i, logL.round(5))
            )
        pi, m, S = maximization(X, g)
        if abs(logL - logL_prev) <= tol:
            break
        logL_prev = logL

    return pi, m, S, g, logL
