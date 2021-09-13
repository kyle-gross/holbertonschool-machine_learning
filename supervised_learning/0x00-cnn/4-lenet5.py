#!/usr/bin/env python3
"""Contains the function lenet5()"""
import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture.
    The model consists of the following layers in order:
        * Convolutional layer with 6 kernels of shape 5x5 with same padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Convolutional layer with 16 kernels of shape 5x5 with valid padding
        * Max pooling layer with kernels of shape 2x2 with 2x2 strides
        * Fully connected layer with 120 nodes
        * Fully connected layer with 84 nodes
        * Fully connected softmax output layer with 10 nodes

    Args:
        x (tf.placeholder)(m,28,28,1): contains input
        y (tf.placeholder)(m,10): contains OH labels

    Returns:
        Tensor for softmax activated output
        Training op that uses Adam opt (w default hyperparameters)
        Tensor for loss
        Tensor for accuracy
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    conv_1 = tf.layers.Conv2D(
        6, (5, 5), padding='same', activation='relu', kernel_initializer=init
    )(x)
    pool_2 = tf.layers.MaxPooling2D(
        (2, 2), (2, 2)
    )(conv_1)
    conv_3 = tf.layers.Conv2D(
        16, (5, 5), activation='relu', kernel_initializer=init
    )(pool_2)
    pool_4 = tf.layers.MaxPooling2D(
        (2, 2), (2, 2)
    )(conv_3)
    pool_4 = tf.layers.Flatten()(pool_4)
    dense_5 = tf.layers.Dense(
        120, activation='relu', kernel_initializer=init
    )(pool_4)
    dense_6 = tf.layers.Dense(
        84, activation='relu', kernel_initializer=init
    )(dense_5)
    y_pred = tf.layers.Dense(
        10, kernel_initializer=init
    )(dense_6)
    output = tf.nn.softmax(y_pred)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    adam = tf.train.AdamOptimizer().minimize(loss)

    return output, adam, loss, accuracy
