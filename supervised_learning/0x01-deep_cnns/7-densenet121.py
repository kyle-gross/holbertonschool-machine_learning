#!/usr/bin/env python3
"""Contains the function densenet121()"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture.

    Args:
        growth_rate (int): growth rate for dense blocks
        compression (float): compression factor for transition layers

    Returns:
        the keras model
    """
    init = K.initializers.he_normal()
    X_input = K.Input((224, 224, 3))

    # Stage 1
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        2*growth_rate, (7, 7), (2, 2), padding='same', kernel_initializer=init
    )(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    X, filters = dense_block(X, 64, growth_rate, 6)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 12)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 24)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 16)

    # AVG POOL
    X = K.layers.AveragePooling2D(
        pool_size=(1, 1), strides=(7, 7), padding='same'
    )(X)

    # Classify
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)

    # Create model
    model = K.Model(inputs=X_input, outputs=X)

    return model
