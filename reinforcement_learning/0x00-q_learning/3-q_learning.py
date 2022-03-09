#!/usr/bin/env python3
"""Contains the function train()"""

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning"""
    totalRewards = []
    maxEpsilon = epsilon

    for episode in range(episodes):
        state = env.reset()

        done = False
        rewardsCurrentEpisode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            
            newState, reward, done, _ = env.step(action)

            Q[state, action] = Q[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(Q[newState, :]))
            
            state = newState
            rewardsCurrentEpisode += reward

            if done == True:
                break

        epsilon = min_epsilon + (maxEpsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        
        totalRewards.append(rewardsCurrentEpisode)
    
    return Q, totalRewards
