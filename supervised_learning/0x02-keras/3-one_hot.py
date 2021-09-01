#!/usr/bin/env python3
"""Contains the function one_hot()"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix

    Args:
        labels (list): integers representing classes
        classes (integer): number of classes

    Returns:
        One-hot matrix
    """
    oh = K.utils.to_categorical(labels, num_classes=classes)

    return oh
