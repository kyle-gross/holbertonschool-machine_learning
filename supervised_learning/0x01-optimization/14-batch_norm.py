#!/usr/bin/env python3
"""Contains the function create_batch_norm_layer()
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in Tensorflow

    Args:
        prev: activated output of previous layer
        n: number of nodes in layer to be created
        actiavtion: activation function that should be used for the output
            of the layer

    Returns:
        Tensor of the activated output for the layer
    """
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8

    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(n, kernel_initializer=kernel)
    mean, variance = tf.nn.moments(base(prev), 0)
    batch_norm = tf.nn.batch_normalization(
        base(prev), mean, variance, beta, gamma, epsilon
    )

    return activation(batch_norm)
