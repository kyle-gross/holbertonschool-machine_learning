#!/usr/bin/env python3
"""Contains the function train_model()"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Trains a Keras model using mini-batch gradient descent

    Args:
        network (keras model): model to train
        data (numpy.ndarray)(m, nx): input data
        labels (numpy.ndarray)(m, classes): contains labels of data
        batch_size (int): size of batch for gradient descent
        epochs (int): number of passes through training set
        validation_data (numpy.ndarray): validation set
        verbose (bool): if True print output
        shuffle (bool): if True, shuffle data between epochs

    Returns:
        generated History object
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle
    )
