#!/usr/bin/env python3
"""Contains the function build_model()"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network using the Keras library without using
    the Sequential() class

    Args:
        nx (int): number of input features
        layers (list): number of nodes used for each layer
        activations (list): activation functions for each layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): dropout parameter

    Returns:
        Keras model
    """
    regularizer = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))

    for i in range(len(layers)):
        dense = K.layers.Dense(
                layers[i],
                activations[i],
                kernel_regularizer=regularizer
            )
        if i == 0:
            x = dense(inputs)
        else:
            x = dense(x)
        if i < (len(layers) - 1):
            x = K.layers.Dropout(1-keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
