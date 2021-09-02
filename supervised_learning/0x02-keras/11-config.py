#!/usr/bin/env python3
"""Contains the functions save_config() and load_config()"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format

    Args:
        network (keras model): model to save config of
        filename (str): path to where file config should be saved

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """Loads a model with specific configuration

    Arg:
        filename (str): path to file to load from

    Returns:
        loaded model
    """
    return K.model.from_json(filename)
