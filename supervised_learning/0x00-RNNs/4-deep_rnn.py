#!/usr/bin/env python3
"""Contains the function deep_rnn()"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN
    Args:
        rnn_cells (list): RNNCell instances of length `l`
            l: no. layers
        X (ndarray)(t,m,i): data to use
            t: max no. time steps
            m: batch size
            i: dimensionality of data
        h_0 (ndarray)(l,m,h): initial hidden state
            h: dimensionality of hidden state
    Returns:
        H, Y
        H (ndarray): contains all of the hidden states
        Y (ndarray): contains all of the outputs
    """
    T, m, i = X.shape
    l, _, h = h_0.shape
    L = len(rnn_cells)

    H = np.zeros((T + 1, l, m, h))
    Y = []
    h_next = h_0

    for t in range(T):
        for i in range(L):
            if i == 0:
                x = X[t]
            else:
                x = H[t + 1, l - 1]
            H[t + 1, l], y = rnn_cells[i].forward(h_next[i], x)
        Y.append(y)
        h_next = H[t + 1]

    return H, np.array(Y)
