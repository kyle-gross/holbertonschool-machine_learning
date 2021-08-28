#!/usr/bin/env python3
"""Contains the function dropout_create_layer()"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using Dropout.

    Args:
        prev (tensor): contains output of prev layer
        n (int): number of nodes for new layer
        activation (tensor): activation function for layer
        keep_prob (float): probability a node will be kept

    Returns:
        Output of new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=(1-keep_prob))
    layer = tf.layers.Dense(
        n, activation, kernel_initializer=init, kernel_regularizer=dropout
    )
    return layer(prev)
