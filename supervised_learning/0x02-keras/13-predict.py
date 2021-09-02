#!/usr/bin/env python3
"""Contains the function predict()"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction

    Args:
        network (keras model): network to predict with
        data (numpy.ndarray): data to predict over
        verbose (bool): if True, prints during prediction process

    Returns:
        the prediction
    """
    return network.predict(
        data,
        verbose=verbose
    )
