#!/usr/bin/env python3
"""Contains the function create_confustion_matrix()
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix

    Args:
        labels: OH numpy.ndarray, shape(m, classes) - correct labels
            * m: number of data points
            * classes: number of classes
        logits: OH numpy.ndarray, shape(m, classes) - precitions

    Returns:
        Confusion numpy.ndarray, shape(classes, classes)
            * row indices represent correct labels
            * column indices represent predicted labels
    """
    confusion_matrix = np.matmul(labels.T, logits)

    return confusion_matrix
