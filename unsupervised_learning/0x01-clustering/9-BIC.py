#!/usr/bin/env python3
"""Contains the function BIC()"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion

    Args:
        X (ndarray)(n,d): data set
        kmin (int): minimum no. clusters to check for
        kmax (int): maximum no. clusters to check for
        iterations (int): maximum no. iterations to use
        tol (float): tolerance for EM algorithm
        verbose (bool): determines if EM algorithm should print to stdout

    Returns:
        best_k, best_result, l, b // None, None, None, None if failure
        best_k (int): best value for k
        best_result (tuple)(pi, m, S):
            pi (ndarray)(k,): cluster priors for best no. clusters
            m (ndarray)(k,d): centroid means for best no. clusters
            S (ndarray)(k,d,d): covariance matrices for best no. clusters
        l (ndarray)(kmax-kmin+1): log likelihood for each cluster size tested
        b (ndarray)(kmax-kmin+1): BIC value for each cluster size tested
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if (type(kmin) is not int or kmin <= 0 or kmin > X.shape[0] or
       type(kmax) is not int or kmax <= 0 or kmax <= kmin or
       kmax > X.shape[0] or
       type(iterations) is not int or iterations <= 0 or
       type(tol) is not float or tol < 0 or
       type(verbose) is not bool):
        return None, None, None, None
