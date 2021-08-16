#!/usr/bin/env python3
"""Contains the function 'create_train_op'.
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates training operation for the network.

    Args:
      loss: loss of the network's prediction
      alpha: learning rate

    Returns:
      operation that trains the network using gradient descent
    """
    gradient = tf.train.GradientDescentOptimizer(alpha)
    return gradient.minimize(loss)
