#!/usr/bin/env python3
"""Contains the function specificity()
"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Specificity: true negative rate (TNR)

    Args:
        confusion: numpy.ndarray, shape(classes,classes) - confusion matrix

    Returns:
        numpy.ndarray, shape(classes,) containing specificity of each class
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    TNR = TN / (TN + FP)

    return TNR
