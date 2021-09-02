#!/usr/bin/env python3
"""Contains the functions save_weights() and load_weights()"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves a model's weights

    Args:
        network (keras model): model whose weights should be saved
        filename (str): path of file the weights should be saved to
        save_format (str): format in which to save the weights

    Returns:
        None
    """
    network.save_weights(
        filename,
        save_format=save_format
    )


def load_weights(network, filename):
    """Loads a model's weights

    Args:
        network (keras model): model to which the weights should
            be loaded
        filename (str): path to file containing weights to load

    Returns:
        None
    """
    network.load_weights(filename)
