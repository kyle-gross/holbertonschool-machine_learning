#!/usr/bin/env python3
"""Contains the function td_lambtha()"""

from turtle import st
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the TD(λ) algorithm"""
    
    for episode in range(episodes):
        state = env.reset()
        eligibility_traces = np.zeros(V.shape[0])

        for j in range(max_steps):
            action = policy(state)
            new_state, reward, done, info = env.step(action)

            eligibility_traces[state] += 1.0
            delta = reward + gamma * V[new_state] - V[state]
            V += alpha * delta * eligibility_traces
            eligibility_traces *= lambtha * gamma

            if done:
                break
            else:
                state = new_state

    return V
