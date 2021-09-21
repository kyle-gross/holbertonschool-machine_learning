#!/usr/bin/env python3
"""Contains the function inception_network()"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds an inception network, GoogLeNet

    Returns:
        the Keras model
    """
    init = K.initializers.he_normal(seed=None)
    input = K.Input(shape=(224, 224, 3))

    # Keras network: number suffix = layer number
    conv0 = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
        activation='relu', kernel_initializer=init
    )(input)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(conv0)

    conv2 = K.layers.Conv2D(
        filters=192, kernel_size=(3, 3), padding='same', activation='relu',
        kernel_initializer=init
    )(pool1)

    pool3 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(conv2)

    inc4 = inception_block(pool3, [64, 96, 128, 16, 32, 32])

    inc5 = inception_block(inc4, [128, 128, 192, 32, 96, 64])

    pool6 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(inc5)

    inc7 = inception_block(pool6, [192, 96, 208, 16, 48, 64])

    inc8 = inception_block(inc7, [160, 112, 224, 24, 64, 64])

    inc9 = inception_block(inc8, [128, 128, 256, 24, 64, 64])

    inc10 = inception_block(inc9, [112, 144, 288, 32, 64, 64])

    inc11 = inception_block(inc10, [256, 160, 320, 32, 128, 128])

    pool12 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(inc11)

    inc13 = inception_block(pool12, [256, 160, 320, 32, 128, 128])

    inc14 = inception_block(inc13, [384, 192, 384, 48, 128, 128])

    avg_pool15 = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1), padding='same'
    )(inc14)

    dropout16 = K.layers.Dropout(0.4)(avg_pool15)

    output = K.layers.Dense(
        1000, activation='softmax', kernel_initializer=init
    )(dropout16)

    model = K.Model(inputs=input, outputs=output)

    return model
