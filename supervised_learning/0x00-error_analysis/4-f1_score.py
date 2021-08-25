#!/usr/bin/env python3
"""Contains the function f1_score()
"""
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix

    Args:
        confusion: numpy.ndarray, shape(classes,classes) - confusion matrix

    Returns:
        numpy.ndarray, shape(classes,) - contains F1 score for each class
    """
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    F1 = 2 * ((PPV * TPR) / (PPV + TPR))

    return F1
