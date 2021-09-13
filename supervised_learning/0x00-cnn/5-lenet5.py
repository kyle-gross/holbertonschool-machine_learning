#!/usr/bin/env python3
"""Contains the function lenet5()"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified verrsion of the LeNet-5 architecture using keras
    The model consists of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
        the he_normal initialization method
    All hidden layers requiring activation should use the relu activation

    Args:
        X (K.input)(m,28,28,1): contains input images

    Returns:
        K.model compiled to use Adam optimization (w default hyperparameters)
        and accuracy metrics.
    """
    init = K.initializers.he_normal(seed=None)
    conv_1 = K.layers.Conv2D(
        6, (5, 5), padding='same', activation='relu', kernel_initializer=init
    )(X)
    pool_2 = K.layers.MaxPool2D(
        strides=(2, 2)
    )(conv_1)
    conv_3 = K.layers.Conv2D(
        16, (5, 5), activation='relu', kernel_initializer=init
    )(pool_2)
    pool_4 = K.layers.MaxPool2D(
        strides=(2, 2)
    )(conv_3)
    pool_4 = K.layers.Flatten()(pool_4)
    dense_5 = K.layers.Dense(
        120, activation='relu', kernel_initializer=init
    )(pool_4)
    dense_6 = K.layers.Dense(
        84, activation='relu', kernel_initializer=init
    )(dense_5)
    y_pred = K.layers.Dense(
        10, activation='softmax', kernel_initializer=init
    )(dense_6)

    opt = K.optimizers.Adam()
    model = K.Model(X, y_pred)
    model.compile(
        opt, loss='categorical_crossentropy', metrics=['accuracy']
    )

    return model
