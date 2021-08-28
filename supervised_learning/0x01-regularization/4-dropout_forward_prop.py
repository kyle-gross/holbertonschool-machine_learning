#!/usr/bin/env python3
"""Contains the function dropout_forward_prop()"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout.

    Args:
        X (numpy.ndarray)(nx, m): contains input data
        weights (dict): contains weights and biases
        L (int): number of layers
        keep_prob (float): probability that a node will be kept

    Returns:
        Dict containing outputs of each layer and the dropout mask used
        on each layer
    """
    output = {}
    output['A0'] = X
    for i in range(L):
        W = weights['W'+str(i+1)]
        b = weights['b'+str(i+1)]
        x = output['A'+str(i)]
        z = np.matmul(W, x) + b
        if i == (L - 1):
            # last layer gets softmax
            act = (np.exp(z)/np.sum(np.exp(z), axis=0))
        else:
            # other layers get tanh
            act = np.tanh(z)
            dropout = np.random.binomial(n=1, p=keep_prob, size=act.shape)
            output['D'+str(i+1)] = dropout
            act = (act * dropout) / keep_prob
        output['A'+str(i+1)] = act
    return output
