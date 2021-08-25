#!/usr/bin/env python3
"""Contains the function precision()
"""
import numpy as np


def precision(confusion):
    """Calculates the precision (PPV) for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray, shape(classes,classes) - confusion matrix

    Returns:
        numpy.ndarray, shape(classes,) - contains precision of each class
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    TP = np.diag(confusion)
    PPV = TP / (TP + FP)

    return PPV
