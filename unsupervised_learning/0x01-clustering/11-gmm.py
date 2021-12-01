#!/usr/bin/env python3
"""Contains the function gmm()"""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset

    Args:
        X (ndarray)(n,d): dataset
        k (int): no. clusters

    Returns:
        pi, m, S, clss, bic
        pi (ndarray)(k,): cluster priors
        m (ndarray)(k,d): centroid means
        S (ndarray)(k,d,d): covariance matrices
        clss (ndarray)(n,): cluster indices for each data point
        bic (ndarray)(kmax-kmin+1): BIC values for each cluster size tested
    """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
