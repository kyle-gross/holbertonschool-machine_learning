#!/usr/bin/env python3
"""Contains the function identity_block()"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block

    Args:
        A_prev (keras input layer): output of prev layer
        filters (tuple/list)(F11,F3,F12):
            F11: no. filters in first 1x1 conv
            F3: no. filters in 3x3 conv
            F12: no. filters in 2nd 1x1 conv

    Returns:
        activated output of identity block
    """
    init = K.initializers.he_normal(seed=None)
    F11, F3, F12 = filters

    # 1x1 conv
    conv0 = K.layers.Conv2D(
        F11, (1, 1), padding='same', activation='linear',
        kernel_initializer=init
    )(A_prev)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv0)
    act2 = K.layers.Activation('relu')(batch_norm1)

    # 3x3 conv
    conv3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='linear',
        kernel_initializer=init
    )(act2)
    batch_norm4 = K.layers.BatchNormalization(axis=3)(conv3)
    act5 = K.layers.Activation('relu')(batch_norm4)

    # 1x1 conv
    conv6 = K.layers.Conv2D(
        F12, (1, 1), padding='same', activation='linear',
        kernel_initializer=init
    )(act5)
    batch_norm7 = K.layers.BatchNormalization(axis=3)(conv6)

    # Add prev activation and original input
    add8 = K.layers.Add()([batch_norm7, A_prev])

    # Activation
    output = K.layers.Activation('relu')(add8)

    return output
