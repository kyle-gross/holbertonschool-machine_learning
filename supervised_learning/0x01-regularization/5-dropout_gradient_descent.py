#!/usr/bin/env python3
"""Contains the function dropout_gradient_descent()"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases of a neural network with Dropout
    regularization using gradient desecent.

    Args:
        Y (numpy.ndarray)(classes,m)(OH): contains labels
        weights (dict): weights and biases of network
        cache (dict): outputs and dropout masks of each layer
        alpha (float): learning rate
        keep_prob (float): probability a node will be kept
        L (int): number of layers

    Returns:
        Nothing, weights and biases are updated in place
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A'+str(i)]
        A_next = cache['A'+str(i-1)]
        if i == L:
            dz = (A - Y)
        else:
            dz_prev = dz
            dropout = cache['D'+str(i)]
            dz = np.matmul(W.T, dz_prev) * (1-(A**2))
            dz = (dz * dropout) / keep_prob
        W = weights['W'+str(i)]
        b = weights['b'+str(i)]
        dW = (1/m) * (np.matmul(dz, A_next.T))
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        weights['W'+str(i)] = W - (alpha * dW)
        weights['b'+str(i)] = b - (alpha * db)
