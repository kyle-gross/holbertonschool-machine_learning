#!/usr/bin/env python3
"""Contains the function 'calculate_accuracy'
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Determines cost of neural network

    Args:
      y: placeholder for labels of input data
      y_pred: tensor containing network's predictions

    Returns:
      tensor containing the decimal accuracy of the prediction
    """
    cost = tf.square(y - y_pred, name='cost')
    return tf.reduce_mean(cost)
