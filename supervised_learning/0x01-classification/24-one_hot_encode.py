#!/usr/bin/env python3
"""
Contains the function one_hot_encode.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    * @Y: numpy.ndarray. Shape: (m,). Contains numeric class labels.
    * @classes: maximum # of classes found in @Y
    Return: one-hot encoding of Y with shape (classes, m) or None if failure.
    """
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    if classes < 2 or classes < np.max(Y):
        return None
    onehot_encode = np.zeros((classes, len(Y)))
    onehot_encode[Y, np.arange(len(Y))] = 1
    return onehot_encode
