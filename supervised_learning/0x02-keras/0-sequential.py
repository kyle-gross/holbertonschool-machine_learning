#!/usr/bin/env python3
"""Contains the function build_model()"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network using Keras library

    Args:
        nx (int): number of features
        layers (list): contains number of nodes in each layer
        activations (list): contains activations for each layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): dropout parameter

    Returns:
        Keras model
    """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))
        if i < (len(layers) - 1):
            # add dropout layers for all but last (output) layer
            model.add(K.layers.Dropout(1-keep_prob))

    return model
