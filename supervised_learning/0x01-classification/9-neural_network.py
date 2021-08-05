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
        Private instance attributes:
            * W1: Weights vector for hidden layer.
            * b1: Bias for hidden layer.
            * A1: Activated output of hidden layer.
            * W2: Weights vector for output neuron.
            * b2: Bias for output neuron.
            * A2: Activated output of output neuron.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        elif nodes < 1:
            raise ValueError('nodes must be a positive integer')
        else:
            self.__W1 = np.random.normal(size=(nodes, nx))
            self.__b1 = np.zeros((nodes, 1))
            self.__A1 = 0
            self.__W2 = np.random.normal(size=(1, nodes))
            self.__b2 = 0
            self.__A2 = 0

    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2
