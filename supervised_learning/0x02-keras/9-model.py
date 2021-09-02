#!/usr/bin/env python3
"""Contains the functions save_model() and load_model()"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves a model

    Args:
        network (keras model): model to save
        filename (str): path to where the model should be saved

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """Loads a model

    Arg:
        filename (str): path from where the model should load

    Returns:
        the loaded model
    """
    return K.models.load_model(filename)
