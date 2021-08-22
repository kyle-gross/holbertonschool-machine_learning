#!/usr/bin/env python3
"""Contains the function update_variables_Adam()
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha: learning rate
        beta1: weight used for first moment
        beta2: weight used for second moment
        epsilon: small number to avoid division by 0
        var: numpy.ndarray - contains variable to be updated
        grad: numpy.ndarray - contains the gradient of var
        v: previous first moment of var
        s: previous second moment of var
        t: time step used for bias correction

    Returns:
        Updated variable, new first moment, new second moment
    """
    vdw = beta1 * v + (1-beta1) * grad
    sdw = beta2 * s + (1-beta2) * grad ** 2
    vdw_corrected = vdw / (1-beta1**t)
    sdw_corrected = sdw / (1-beta2**t)
    w = var - alpha * (vdw_corrected / (np.sqrt(sdw_corrected) + epsilon))

    return w, vdw, sdw
