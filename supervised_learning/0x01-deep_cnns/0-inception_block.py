#!/usr/bin/env python3
"""Contains the function inception_block()"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block.

    Args:
        A_prev (keras input layer): output of prev layer
        filters (tuple/list)(F1,F3R,F3,F5R,F5,FPP):
            F1: no. filters in 1x1 convolution
            F3R: no. filters in 1x1 convolution before the 3x3 convolution
            F3: no. filters in 3x3 convolution
            F5R: no. filters in 1x1 convolution before 5x5 convolution
            F5: no. filters in 5x5 convolution
            FPP: no. filters in 1x1 convolution after max pooling

    Returns:
        concatenated output of inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    # 1x1 conv
    conv1 = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(A_prev)

    # 1x1 conv into 3x3 conv
    conv3R = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(A_prev)
    conv3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu', kernel_initializer=init
    )(conv3R)

    # 1x1 conv into 5x5 conv
    conv5R = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(A_prev)
    conv5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu', kernel_initializer=init
    )(conv5R)

    # 3x3 max pooling into 1x1 conv
    pool = K.layers.MaxPooling2D(
        (2, 2), strides=(1, 1), padding='same'
    )(A_prev)
    conv1x = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu', kernel_initializer=init
    )(pool)

    # Concatenate filters
    layer_out = K.layers.concatenate([conv1, conv3, conv5, conv1x])

    return layer_out
