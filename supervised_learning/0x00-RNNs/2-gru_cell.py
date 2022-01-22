#!/usr/bin/env python3
"""Contains the class GRUCell"""

import numpy as np


class GRUCell():
    """Gated recurrent unit"""
    def __init__(self, i, h, o):
        """Instantiates a GRUCell object
        Args:
            i: dimensionality of the data
            h: dimensionality of hidden state
            o: dimensionality of the outputs
        Attributes:
            Wz & bz: update gate
            Wr & br: reset gate
            Wh & bh: intermediate hidden state
            Wy & by: output
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(z):
        """Softmax activation function"""
        return np.exp(z) / (np.sum(np.exp(z), axis=1, keepdims=True))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step
        Args:
            x_t (ndarray)(m,i): input data
                m: batch size
            h_prev (ndarray)(m,h): previous hidden state
        Return:
            h_next, y
                h_next: next hidden state
                y: output of the cell
        """
        m, i = x_t.shape

        input = np.concatenate((h_prev, x_t), axis=1)
        # Reset gate
        r_t = self.sigmoid(np.matmul(input, self.Wr) + self.br)

        # Update gate
        z_t = self.sigmoid(np.matmul(input, self.Wz) + self.bz)

        r_prev = r_t * h_prev
        r_input = np.concatenate((r_prev, x_t), axis=1)

        h_t = np.tanh(np.matmul(r_input, self.Wh) + self.bh)
        h_next = ((1 - z_t) * h_prev) + (z_t * h_t)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
