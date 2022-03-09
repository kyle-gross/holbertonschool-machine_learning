#!/usr/bin/env python3
"""Contains the function epsilon_greedy()"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)

    if p < epsilon:
        nextAction = np.random.randint(Q.shape[1])
    else:
        nextAction = np.argmax(Q[state, :])

    return nextAction
