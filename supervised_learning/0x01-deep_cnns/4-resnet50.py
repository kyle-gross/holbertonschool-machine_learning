#!/usr/bin/env python3
"""Contains the function resnet50()"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture.

    Returns:
        the keras model
    """
    init = K.initializers.he_normal(seed=None)

    X_input = K.Input(shape=(224, 224, 3))
    X = K.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', activation='linear',
        kernel_initializer=init
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    filters = [64, 64, 256]
    X = projection_block(X, filters, s=1)
    X = identity_block(X, filters)
    X = identity_block(X, filters)

    # Stage 3
    filters = [128, 128, 512]
    X = projection_block(X, filters, s=2)
    X = identity_block(X, filters)
    X = identity_block(X, filters)
    X = identity_block(X, filters)

    # Stage 4
    filters = [256, 256, 1024]
    X = projection_block(X, filters, s=2)
    X = identity_block(X, filters)
    X = identity_block(X, filters)
    X = identity_block(X, filters)
    X = identity_block(X, filters)
    X = identity_block(X, filters)

    # Stage 5
    filters = [512, 512, 2048]
    X = projection_block(X, filters, s=2)
    X = identity_block(X, filters)
    X = identity_block(X, filters)

    # AVGPOOL
    X = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X)

    # Output layer
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
    )(X)

    # Create model
    model = K.Model(inputs=X_input, outputs=X)

    return model
