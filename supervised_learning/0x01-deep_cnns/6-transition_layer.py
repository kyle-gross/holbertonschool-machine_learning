#!/usr/bin/env python3
"""Contains the function transition_layer()"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer.

    Args:
        X (keras input): output of prev layer
        nb_filters (int): no. filters in X
        compression (float): compression factor for transition layer
    
    Returns:
        Output of the transition layer and no. filters within output,
        respectively.
    """
    init = K.initializers.he_normal()
    filters = int(compression * nb_filters)

    x = K.layers.BatchNormalization(axis=3)(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        filters, (1, 1), padding='same', kernel_initializer=init
    )(x)
    x = K.layers.AveragePooling2D((2, 2))(x)

    return x, filters
