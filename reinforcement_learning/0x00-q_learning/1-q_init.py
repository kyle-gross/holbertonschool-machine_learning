#!/usr/bin/env python3
"""Contains the function q_init()"""

import numpy as np


def q_init(env):
    """Initializes the Q-table"""
    actionSpaceSize = env.action_space.n
    stateSpaceSize = env.observation_space.n

    q_table = np.zeros((stateSpaceSize, actionSpaceSize))

    return q_table
