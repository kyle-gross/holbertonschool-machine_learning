#!/usr/bin/env python3
"""Contains the function sensitivity()
"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity (recall) for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray, shape(classes,classes) - confusion matrix

    Returns:
        numpy.ndarray, shape(classes,) - contains sensitivity of each class
    """
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TPR = TP / (TP + FN)

    return TPR
