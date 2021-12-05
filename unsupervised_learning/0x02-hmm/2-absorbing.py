#!/usr/bin/env python3
"""Contains the function absorbing()"""

import numpy as np


def get_to_abs_state(keys, i, P):
    """Determines if a state leads to an absorbing state or not"""
    X = P.T[i]
    # Append states that will lead to absorbing state
    [keys.append(j) for j in range(P.shape[0]) if X[j] > 0]

    return keys


def absorbing(P):
    """Determines if a markov chain is absorbing

    Args:
        P (ndarray)(n,n): standard trasition matrix
            P[i,j]: probability of transitioning from state i to state j
            n: no. states in the markov chain

    Returns:
        True if absorbing // False if failure
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1]):
        return False

    diag = np.diag(P)
    if not np.any(diag == 1):
        return False
    if np.all(diag == 1):
        return True

    keys = [i for i in range(len(diag)) if diag[i] == 1]
    
    for i in range(P.shape[0]):
        if i in keys:
            keys = get_to_abs_state(keys, i, P)

    return len(set(keys)) == P.shape[0]
