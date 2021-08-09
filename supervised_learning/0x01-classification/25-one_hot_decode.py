#!/usr/bin/env python3
"""
Contains the function one_hot_decode().
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.
    * @one_hot: one-hot encoded numpy.ndarray. Shape: (classes, m)
        * classes: max # of classes
        * m: number of examples
    Return: numpy.ndarray. Shape: (m,). Contains numeric labels for
            each example. Or None if failure
    """
    if type(one_hot) is not np.ndarray:
        return None
    if one_hot.ndim != 2:
        return None
    return np.argmax(one_hot, axis=0)
