#!/usr/bin/env python3
"""Contains the function bi_rnn()"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN
    Args:
        bi_cell: BidirectionalCell object used for forward prop
        X (ndarray)(t,m,i): input data
            t: max no. time steps
            m: batch size
            i: dimensionality of the data
        h_0 (ndarray)(m,h): initial forward hidden state
            h: dimensionality of hidden state
        h_t (ndarray)(m,h): backward hidden state
    Returns:
        H, Y
        H (ndarray): concatenated hidden states
        Y (ndarray): all outputs
    """
    T, m, _ = X.shape
    _, h = h_0.shape
    H = np.zeros((T, m, h * 2))
    H_F = np.zeros((T, m, h))
    H_B = np.zeros((T, m, h))

    h_prev = h_0
    h_next = h_t

    for t in range(T):
        H_F[t] = bi_cell.forward(h_prev, X[t])
        H_B[T - t - 1] = bi_cell.backward(h_next, X[T - t - 1])
        h_next = H_B[T - t - 1]
        h_prev = H_F[t]
    H = np.concatenate((H_F, H_B), axis=-1)
    Y = bi_cell.output(H)

    return H, np.array(Y)
