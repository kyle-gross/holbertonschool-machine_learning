#!/usr/bin/env python3
"""Contains the function optimize_model()"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a Keras model with categorical
    crossentropy loss and accuracy metrics

    Args:
        network (keras model): model to optimize
        alpha (float): learning rate
        beta1 (float): first Adam parameter
        beta2 (float): second Adam parameter

    Returns:
        None
    """
    opt = K.optimizers.Adam(
        lr=alpha,
        beta_1=beta1,
        beta_2=beta2
    )
    network.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
