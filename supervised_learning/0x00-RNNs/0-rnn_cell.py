#!/usr/bin/env python3
"""Contains the class RNNCell"""

import numpy as np


class RNNCell():
    """Represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """Instantiates an RNNCell object

        Args:
            i: dimensionality of the data
            h: dimensionality of hidden state
            o: dimensionality of the outputs

        Attributes:
            Wh & bh: weights and biases of the cell
            Wy & by: output
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        """Softmax activation function"""
        return np.exp(z) / (np.sum(np.exp(z), axis=1, keepdims=True))

    def forward(self, h_prev, x_t):
        """Performs forward propgation for one time step

        Args:
            h_prev (ndarray)(m,h): previous hidden state
                m: batch size for the data
            x_t (ndarray)(m,i): data input for the cell
        """
        dot_x = np.dot(x_t, self.Wh[h_prev.shape[1]:, :])
        dot_h = np.dot(h_prev, self.Wh[:h_prev.shape[1], :])
        h_next = np.tanh(dot_x + dot_h + self.bh)

        return h_next, self.softmax(np.dot(h_next, self.Wy) + self.by)
