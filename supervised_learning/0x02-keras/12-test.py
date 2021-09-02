#!/usr/bin/env python3
"""Contains the function test_model()"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network

    Args:
        network (keras model): network to test
        data (numpy.ndarray): input data
        labels (numpy.ndarray): OH matrix of labels
        verbose (bool): if True, print output during testing

    Returns:
        loss and accuracy of model with testing data, respectively
    """
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
