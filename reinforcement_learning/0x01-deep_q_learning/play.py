#!/usr/bin/env python3
"""Loads a trained DQNAgent and plays Atari Breakout"""

import gym
from tensorflow.keras.optimizers import Adam
from train import createQModel, createAgent, ENV_NAME


def play():
    """Loads a trained DQNAgent and has it play 10 rounds of Atari Breakout."""
    # Setup environment
    env = gym.make(ENV_NAME) # render_mode='human'
    actions = env.action_space.n

    # Create model & agent
    model = createQModel(actions)
    dqn = createAgent(model, actions)

    # Compile and load weights
    dqn.compile(optimizer=Adam(learning_rate=1e-7))
    dqn.load_weights('./policy.h5')

    # Play
    dqn.test(env, nb_episodes=10, visualize=True)
    env.close()


if __name__ == '__main__':
    play()