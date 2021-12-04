#!/usr/bin/env python3
"""Contains the function regular()"""

import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain

    Args:
        P (ndarray)(n,n): transition matrix
            P[i,j]: probability of transitioning from state i to state j
            n: no. states in markov chain

    Returns:
        steady // None if failure
        steady (ndarray)(1,n): steady state probabilities
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1] or not (P > 0).all()):
        return None

    evals, evecs = np.linalg.eig(P.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    steady = evec1 / evec1.sum()
    
    return np.array([steady])
