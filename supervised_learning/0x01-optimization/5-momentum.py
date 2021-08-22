#!/usr/bin/env python3
"""Contains the function update_variables_momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent with
    momentum algorithm.

    Args:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray - contains variable to be updated
        grad: numpy.ndarray - contains the gradient of var
            * dw
        v: previous first moment of var
            * prev dw

    Returns:
        Updated variable and the new moment, respectively
    """
    vdw = beta1 * v + (1 - beta1) * grad
    w = var - alpha * vdw

    return w, vdw
