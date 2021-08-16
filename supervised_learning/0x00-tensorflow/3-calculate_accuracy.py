#!/usr/bin/env python3
"""Contains the function 'calculate_accuracy'
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Determines accuracy of neural network

    Args:
      y: placeholder for labels of input data
      y_pred: tensor containing network's predictions

    Returns:
      tensor containing the decimal accuracy of the prediction
    """
    y = tf.argmax(y)
    y_pred = tf.argmax(y_pred)
    return tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
