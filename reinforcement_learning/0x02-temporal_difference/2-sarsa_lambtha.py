#!/usr/bin/env python3
"""Contains the function sarsa_lambtha()"""

import numpy as np


def epsilon_greedy(env, Q, state, epsilon):
    """Performs epsilon-greedy"""
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs the SARSA(λ) algorithm"""
    max_epsilon = epsilon
    eligibility_traces = np.zeros(Q.shape)

    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(env, Q, state, epsilon)

        for j in range(max_steps):
            new_state, reward, done, info = env.step(action)

            new_action = epsilon_greedy(env, Q, state, epsilon)

            eligibility_traces *= gamma * epsilon
            eligibility_traces[state, action] += 1.0

            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
            Q += alpha * delta * eligibility_traces

            if done:
                break
            else:
                state = new_state
                action = new_action
        
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon *= max_epsilon * np.exp(-epsilon_decay * i)

    return Q
