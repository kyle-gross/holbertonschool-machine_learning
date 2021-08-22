#!/usr/bin/env python3
"""Contains the function update_variables_RMSProp()
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero
        var: numpy.ndarray - contains variable to update
        grad: numpy.ndarray - contains gradient of var
            * dw
        s: previous second moment of var
            * dw_prev

    Returns:
        Updated variable and new moment, respectively
    """
    sdw = beta2 * s + (1-beta2) * grad ** 2
    w = var - alpha * (grad / (np.sqrt(sdw) + epsilon))

    return w, sdw
