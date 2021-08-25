#!/usr/bin/env python3
"""Contains the function f1_score()
"""
import numpy as np


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix

    Args:
        confusion: numpy.ndarray, shape(classes,classes) - confusion matrix

    Returns:
        numpy.ndarray, shape(classes,) - contains F1 score for each class
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    F1 = (2 * TP) / (2 * TP + FP + FN)

    return F1
