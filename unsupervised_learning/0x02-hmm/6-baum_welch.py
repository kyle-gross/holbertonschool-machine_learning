#!/usr/bin/env python3
"""Contains the function baum_welch()"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model

    Args:
        Observations (ndarray)(T,): index of the observation
            T: no. observations
        Transition (ndarray)(M,M): transition matrix
            M: no. hidden states
        Emission (ndarray)(M,N): emission probabilities
            N: no. output states
        Initial (ndarray)(M,1): starting probabilities
        Iterations (int): no. times EM should be performed

    Returns:
        Converged Transition, Emission // None, None if failure
    """
    return None, None
