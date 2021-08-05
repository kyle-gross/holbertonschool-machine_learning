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

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron.
        * @X: numpy.ndarray with shape (nx, m). Contains input data.
            * nx: number of input features
            * m: number of examples
        * Updates private attribute __A. Uses a sigmoid activation function
        Return: __A
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates cost of the model using logistic regression.
        * @Y: numpy.ndarray with shape (1, m).
            * Contains correct labels for input data.
        * @A: numpy.ndarray with shape (1, m).
            * Contains activated output of neuron for each example.
        Return: cost (numpy.ndarray)
        """
        m = A.shape[1]
        return -(1/m)*(np.sum((Y*np.log(A))+(1.0000001-Y)*np.log(1.0000001-A)))

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.
        * @X: numpy.ndarray with shape (nx, m). Contains input data.
            * nx: number of input features
            * m: number of examples
        * @Y: numpy.ndarray with shape (1, m). Contains correct labels.
        Return: neuron's prediction and cost of the network.
            * numpy.ndarray with shape (1, m) containing predicted labels
              for each example.
            * Labels should be 1 if output of network is >= 0.5, otherwise 0.
        """
        output = self.forward_prop(X)
        cost = self.cost(Y, output)
        output = np.where(output >= 0.5, 1, 0)
        return output.astype(int), cost
