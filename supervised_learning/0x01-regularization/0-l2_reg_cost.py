#!/usr/bin/env python3
"""Contains the function l2_reg_cost()"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: original cost function
        lambtha: regularization parameter
        weights: dict of weights and biases of the network
        L: number of layers in network
        m: number of data points used

    Returns:
        Cost of the network with L2 regularization
    """
    w_norm = 0
    for layer in range(L):
        w = weights['W{}'.format(layer+1)]
        w_norm += np.linalg.norm(w)

    return cost + (lambtha/(2*m)) * w_norm
