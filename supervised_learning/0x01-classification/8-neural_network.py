#!/usr/bin/env python3
"""
Contains the class NeuralNetwork.
"""
import numpy as np


class NeuralNetwork():
    """
    Defines a neural network with one hidden layer performing
    binary classification.
    """
    def __init__(self, nx, nodes):
        """
        @nx: number of input features.
            * Must be int. Must be > 0.
        @nodes: number of nodes found in the hidden layer.
            * Must be int. Must be > 0.
        Public instance attributes:
            * W1: Weights vector for hidden layer.
            * b1: Bias for hidden layer.
            * A1: Activated output of hidden layer.
            * W2: Weights vector for output neuron.
            * b2: Bias for output neuron.
            * A2: Activated output of output neuron.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
