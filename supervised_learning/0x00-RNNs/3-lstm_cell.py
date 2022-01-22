#!/usr/bin/env python3
"""Contains the class LSTMCell"""

import numpy as np


class LSTMCell():
    """LSTMCell class"""
    def __init__(self, i, h, o):
        """Instantiates an LSTMCell object
        Args:
            i: dimensionality of data
            h: dimensionality of hidden state
            o: dimensionality of outputs
        Attributes:
            Wf & bf: forget gate
            Wu & bu: update gate
            Wc & bc: intermediate cell state
            Wo & bo: output gate
            Wy & by: outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(z):
        """Softmax activation function"""
        return np.exp(z) / (np.sum(np.exp(z), axis=1, keepdims=True))

    def forward(self, h_prev, c_prev, x_t):
        """Forward propagation"""
        combo = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        forget = self.sigmoid(np.dot(combo, self.Wf) + self.bf)
        # Update gate
        update = self.sigmoid(np.dot(combo, self.Wu) + self.bu)
        # Candidate
        candidate = np.tanh(np.dot(combo, self.Wc) + self.bc)
        # Output gate
        output = self.sigmoid(np.dot(combo, self.Wo) + self.bc)
        c_next = (forget * c_prev) + (update * candidate)
        h_next = output * np.tanh(c_next)

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
