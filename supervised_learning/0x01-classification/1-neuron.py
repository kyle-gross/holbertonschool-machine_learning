#!/usr/bin/env python3
"""
Contains the class Neuron.
"""
import numpy as np


class Neuron():
    """
    Defines a single neuron performing binary classification.
    """
    def __init__(self, nx):
        """
        * @nx: number of input features to the neuron
            * Must be int
            * Must be > 0
        * Private instance attributes:
            * __W: weights vector for neuron. Initialized using a random
                   normal distribution.
            * __b: bias for the neuron. Initialized to 0.
            * __A: activated output of neuron (prediction). Initialized
                   to 0.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.normal(size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A
