#!/usr/bin/env python3
"""Contains the function dense_block()"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block.

    Args:
        X (keras input): output of prev layer
        nb_filters (int): number of filters in X
        growth_rate (int): growth rate for dense block
        layers (int): no. layers in dense block

    Returns:
        Concatenated output of each layer within the dense block and
        no. filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()

    for _ in range(layers):
        x = K.layers.BatchNormalization(axis=3)(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            4 * growth_rate, (1, 1), padding='same', kernel_initializer=init
        )(x)
        x = K.layers.BatchNormalization(axis=3)(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same', kernel_initializer=init
        )(x)
        X = K.layers.concatenate([X, x])
        nb_filters += growth_rate

    return X, nb_filters
