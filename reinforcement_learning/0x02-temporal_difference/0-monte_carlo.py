#!/usr/bin/env python3
"""Contains the function monte_carlo()"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the monte carlo algorithm"""

    for i in range(episodes):
        state = env.reset()
        episode = []

        for j in range(max_steps):
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            episode.append([state, reward])
            if done:
                break
            state = new_state
        
        episode = np.array(episode, dtype=int)
        G = 0

        for j, step in enumerate(episode[::-1]):
            state, reward = step
            G = gamma * G + reward
            if state not in episode[:i, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
