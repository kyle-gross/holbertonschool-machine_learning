#!/usr/bin/env python3
"""Contains the function viterbi()"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a hidden
    markov model

    Args:
        Observation (ndarray)(T,): contains the index of the observation
            T: no. observations
        Emission (ndarray)(N,M): contains the emission probability of a
            specific observation given a hidden state
            Emission[i,j]: probability of observing j given the hidden state i
            N: no. hidden states
            M: no. all possible observations
        Transition (ndarray)(N,N): transition matrix
            Transition[i,j]: probability of transitioning from the hidden
                state i to j
        Initial (ndarray)(N,1): probability of starting in a particular hidden
            state

    Returns:
        path, P // None, None if failure
        path (list): length of T - contains most likely sequence of hidden
            states
        P (float): probability of obtaining the path sequence
    """
    if (type(Observation) is not np.ndarray or Observation.ndim != 1 or
       Observation.shape[0] == 0 or
       type(Emission) is not np.ndarray or Emission.ndim != 2 or
       type(Transition) is not np.ndarray or Transition.ndim != 2 or
       type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]):
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape

    V = np.zeros((N, T))
    B = np.zeros((N, T))

    for t in range(len(Observation)):
        for s in range(N):
            if t == 0:
                V[s, 0] = Initial[s, 0] * Emission[s, Observation[0]]
            else:
                temp = (
                    V[:, t-1] * Transition[:, s] * Emission[s, Observation[t]]
                )
                V[s, t] = np.max(temp)
                B[s, t] = np.argmax(temp)

    P = np.max(V[:, T-1])
    pointer = np.argmax(V[:, T-1])
    path = [pointer]

    for t in range(T - 1, 0, -1):
        p = int(B[pointer, t])
        path.append(p)
        pointer = p

    return path[::-1], P
