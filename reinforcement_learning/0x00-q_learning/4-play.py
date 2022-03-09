#!/usr/bin/env python3
"""Contains the function play()"""

import numpy as np


def play(env, Q, max_steps=100):
    """Function which has the trained agent play an episode"""
    state = env.reset()
    done = False

    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state,:])
        state, reward, done, _ = env.step(action)

        if done:
            env.render()
            break
    
    return reward
