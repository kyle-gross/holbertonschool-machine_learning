#!/usr/bin/env python3
"""Contains the function maximization()"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM

    Args:
        X (ndarray)(n,d): contains data set
        g (ndarray)(k,n): contains the posterior probabilities for each data
            point in each cluster

    Returns:
        pi, m, S // None, None, None if failure
        pi (ndarray)(k,): contains updated priors for each cluster
        m (ndarray)(k,d): contains updated centroid means for each cluster
        S (ndarray)(k,d,d): contains updated covariance matrices for each
            cluster
    """
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(g) is not np.ndarray or g.ndim != 2 or
       X.shape[0] != g.shape[1] or
       not np.isclose(np.sum(g, axis=0), np.ones(X.shape[0],)).all()):
        return None, None, None

    k, n = g.shape
    d = X.shape[1]

    pi = g.sum(axis=1)
    pi /= n
    m = np.dot(g, X)
    m /= g.sum(1)[:, None]
    S = np.zeros((k, d, d))

    for i in range(k):
        ys = X - m[i, :]
        S[i] = (
            g[i, :, None, None] * np.matmul(ys[:, :, None], ys[:, None, :])
        ).sum(axis=0)
    S /= g.sum(axis=1)[:, None, None]

    return pi, m, S
