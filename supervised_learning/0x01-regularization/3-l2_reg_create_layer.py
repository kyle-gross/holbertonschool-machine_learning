#!/usr/bin/env python3
"""Contains the function l2_reg_create_layer()"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a Tensorflow layer that includes L2 regularization

    Args:
        prev (tensor): cotains output of prev layer
        n (int): number of nodes of new layer
        activation (tensor): activation function to use on layer
        lambtha: L2 regularization parameter

    Returns:
        Output of new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        n, activation, kernel_initializer=init, kernel_regularizer=regularizer
    )
    return layer(prev)
