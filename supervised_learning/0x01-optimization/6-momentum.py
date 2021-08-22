#!/usr/bin/env python3
"""Contains the function create_momentum_op
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm

    Args:
        loss: loss of network
        alpha: learning rate
        beta1: momentum weight

    Returns:
        Momentum optimization operation
    """
    momentum = tf.train.MomentumOptimizer(alpha, beta1)

    return momentum.minimize(loss)
