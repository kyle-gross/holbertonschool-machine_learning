#!/usr/bin/env python3
"""Contains the function 'calculate_loss'.
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction.

    Args:
      y: placeholder for labels of input data
      y_pred: tensor containing the networks predictions

    Returns:
      tensor containing the loss of the prediction
    """
    return tf.losses.mean_squared_error(labels=y, predictions=y_pred)
