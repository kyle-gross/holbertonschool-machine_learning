#!/usr/bin/env python3
"""Contains the function l2_reg_cost()"""
import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost (tensor): contains cost of network without L2 regularization

    Returns:
        A tensor containing the cost of the network accounting for L2
        regularization
    """
    return cost + tf.losses.get_regularization_losses()
