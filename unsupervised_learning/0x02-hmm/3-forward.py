#!/usr/bin/env python3
"""Contains the function forward()"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model

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
        P, F // None, None if failure
        P: likelihood of the observations given the model
        F (ndarray)(N,T): forward path probabilities
            F[i,j]: probability of being in hidden state i at time j given the
                previous observations
    """
    if (type(Observation) is not np.ndarray or Observation.ndim != 1 or
       Observation.shape[0] == 0 or
       type(Emission) is not np.ndarray or Emission.ndim != 2 or
       type(Transition) is not np.ndarray or Transition.ndim != 2 or
       type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]):
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    # Initialize forward path probabilities to 0
    F = np.zeros((N, T))

    # Loop through observations
    for t in range(len(Observation)):
        # Update forward path prob for each hidden state
        for s in range(N):
            # First observation
            if t == 0:
                F[s, 0] = Initial[s, 0] * Emission[s, Observation[t]]
            # Other observations
            else:
                F[s, t] = np.sum(
                    F[:, t-1] * Transition[:, s] * Emission[s, Observation[t]]
                )

    P = np.sum(F[:, -1])

    return P, F
