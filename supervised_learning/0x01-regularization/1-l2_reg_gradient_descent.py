#!/usr/bin/env python3
"""Contains the function l2_reg_gradient_descent()"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases of a neural network using gradient
    descent with L2 regularization

    Args:
        Y (numpy.ndarray)(classes, m): OH - contains labels
        weights (dict): weights and biases of neural network
        cache (dict): outputs of each layer of network
        alpha (float): learning rate
        lambtha (float): L2 regularization parameter
        L (int): number of layers

    Returns:
        Nothing, weights and biases are updated in place (weights dict)
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A'+str(i)]
        A_next = cache['A'+str(i-1)]
        if i == L:
            dz = (A - Y)
        else:
            dz_prev = dz
            dz = np.matmul(W.T, dz_prev) * (1-(A**2))
        W = weights['W'+str(i)]
        b = weights['b'+str(i)]
        dW = (1/m) * (np.matmul(dz, A_next.T)) + ((lambtha/m)*W)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        W -= (alpha * dW)
        b -= (alpha * db)
