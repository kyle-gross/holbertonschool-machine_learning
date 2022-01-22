#!/usr/bin/env python3
"""Contains the function rnn()"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN

    Args:
        rnn_cell: instance of RNNCell
        X (ndarray)(t,m,i): data
            t: max no. time steps
            m: batch size
            i: dimensionality of data
        h_0 (ndarray)(m,h): initial hidden state
            h: dimensionality of hidden state

    Returns:
        H, Y
        H (ndarray): contains all hidden states
        Y (ndarray): contains all outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = []

    h_next = h_0
    for i, x in enumerate(X):
        H[i + 1], y = rnn_cell.forward(h_next, x)
        h_next = H[i + 1]
        Y.append(y)

    return H, np.array(Y)
