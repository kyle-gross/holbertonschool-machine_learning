#!/usr/bin/env python3
"""Contains the function expectation()"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM

    Args:
        X (ndarray)(n,d): contains data set
        pi (ndarray)(k,): contains priors for each cluster
        m (ndarray)(k,d): contains the centroid means for each cluster
        S (ndarray)(k,d,d): contains covariance matrices for each cluster

    Returns:
        g, l // None, None if failure
            g (ndarray)(k,n): contains the posterior probabilities for each
                data point in each cluster
            l (float): total log likelihood
    """
    if (type(X) is not np.ndarray or X.ndim != 2 or
       type(pi) is not np.ndarray or pi.ndim != 1 or
       type(m) is not np.ndarray or m.ndim != 2 or
       type(S) is not np.ndarray or S.ndim != 3):
        return None, None

    k = pi.shape[0]

    pdfs = [pdf(X, m[i], S[i]) * pi[i] for i in range(k)]

    if any(pdfs) is None:
        return None, None

    g = np.array(pdfs)  # Probabilities
    likelihood = g.sum(axis=0)
    g /= likelihood
    loglikelihood = np.sum(np.log(likelihood))

    return g, loglikelihood
