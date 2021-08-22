#!/usr/bin/env python3
"""Contains the function create_Adam_op()
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in Tensorflow
    using the Adam optimization algorithm.

    Args:
        loss: loss of network
        alpha: learning rate
        beta1: weight used for first moment
        beta2: weight used for second moment
        epsilon: small number to avoid division by 0

    Returns:
        Adam optimization operation
    """
    opt = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)

    return opt.minimize(loss)
