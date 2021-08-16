#!/usr/bin/env python3
"""
Contains the function create_layer.
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer of neurons

    Args:
      prev: tensor output of the previous layer
      n: number of nodes in the layer to create
      activation: activation function that the layer should use

    Returns:
      tensor output of the layer
    """
    theta = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n, activation, kernel_initializer=theta)
