#!/usr/bin/env python3
"""This module contains the classes and functions for training a Deep-Q Agent
to play the game: Atari Breakout."""

import gym
from keras.layers import Activation, Dense, Flatten, Conv2D, Permute
from keras.models import Sequential
import numpy as np
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.optimizers import Adam


ENV_NAME = 'ALE/Breakout-v5'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        """Process the observation
        * Resize to 84 x 84
        * Convert to grayscale
        * Crop irrelevant image parts
        """
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize((INPUT_SHAPE)).convert('L') # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        processed_observation = processed_observation.astype('uint8')
        
        return processed_observation

    def process_state_batch(self, batch):
        """Process state
        We could perform this processing step in `process_observation`.
        In this case, however, we would need to store a `float32`
        array instead, which is 4x more memory intensive than
        an `uint8` array. This matters if we store 1M observations.
        """
        processed_batch = batch.astype('float32') / 255.

        return processed_batch

    def process_reward(self, reward):
        """Process reward"""
        reward = np.clip(reward, -1, 1.)

        return reward


def createQModel(actions):
    """Network defined by Deepmind paper"""
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))

    return model


def createAgent(model, actions):
    """Creates the DQNAgent"""
    processor = AtariProcessor()
    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=0.1,
        value_test=0.05,
        nb_steps=100000
    )
    agent = DQNAgent(
        model=model,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_actions=actions,
        nb_steps_warmup=5000,
        gamma=0.99,
        train_interval=4,
        delta_clip=1.0,
        target_model_update=5000
    )

    return agent


def train():
    """Trains a DQNAgent to play Atari Breakout"""
    # Create environment
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    actions = env.action_space.n

    # Create model and DQNAgent
    model = createQModel(actions)
    dqn = createAgent(model, actions)
    dqn.compile(Adam(learning_rate=1e-7), metrics=['mae'])

    # Train model and save weights
    dqn.fit(env, nb_steps=4000000, verbose=1, log_interval=25000)
    dqn.save_weights('policy.h5', overwrite=True)

    env.close()


if __name__ == '__main__':
    train()
