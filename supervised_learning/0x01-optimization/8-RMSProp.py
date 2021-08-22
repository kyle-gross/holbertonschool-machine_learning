#!/usr/bin/env python3
"""Contains the function create_RMSProp_op()
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm.

    Args:
        loss: loss of network
        aplha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero

    Returns:
        RMSProp optimization operation
    """
    opt = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)

    return opt.minimize(loss)
