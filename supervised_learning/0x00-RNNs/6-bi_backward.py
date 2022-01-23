#!/usr/bin/env python3
"""Contains the class BidirectionalCell"""

import numpy as np


class BidirectionalCell():
    """Represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """Instantiates a BidirectionalCell object
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs
        Attributes:
            Whf & bhf: hidden states forward
            Whb & bhb: hidden states backward
            Wy & by: outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h * 2, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
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
            h_prev (ndarray)(m,i): data input for the cell
                m: batch size
            h_prev (ndarray)(m,h): previous hidden state
        Returns:
            h_next: the next hidden state
        """
        m, i = x_t.shape
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """Calculates hidden state in backward direction for one time step
        Args:
            x_t (ndarray)(m,i): data input for the cell
                m: batch size
            h_next (ndarray)(m,h): next hidden state
        Returns:
            h_prev: previous hidden state
        """
        m, i = x_t.shape
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev
