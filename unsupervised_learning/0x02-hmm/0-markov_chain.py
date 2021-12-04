#!/usr/bin/env python3
"""Contains the function markov_chain()"""

import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chaing being in a particular
    state after a specified number of iterations

    Args:
        P (ndarray)(n,n): transition matrix
            P[i,j]: probability of transitioning from state i to state j
            n: no. states in markov chain
        s (ndarray)(1,n): probability of starting in each state
        t (int): no. iterations that the markov chain has been through

    Returns:
        states // None if failure
        states (ndarray)(1,n): probability of being in a specific state after
            't' iterations.
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
       P.shape[0] != P.shape[1] or
       type(s) is not np.ndarray or s.ndim != 2 or
       s.shape[0] != 1 or s.shape[1] != P.shape[0]):
        return None

    states = np.dot(s, P)

    for _ in range(t):
        states = np.dot(states, P)

    return states
