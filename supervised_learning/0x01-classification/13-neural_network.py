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

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network.
        * @X: numpy.ndarray. Shape: (nx, m). Contains input data.
        * Updates private attributes __A1 and __A2
        * Uses sigmoid activation function.
        Return: __A1 and __A2
        """
        z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-z))
        z = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-z))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates cost of model using logistic regression.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        * @A: numpy.ndarray. Shape: (1, m). Contains activated output.
        Return: cost
        """
        m = A.shape[1]
        return -(1/m)*(np.sum((Y*np.log(A))+(1-Y)*np.log(1.0000001-A)))

    def evaluate(self, X, Y):
        """
        Evaluates neural network's predictions.
        * @X: numpy.ndarray. Shape: (nx, m). Contains input data.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        Return: Neuron's prediction and cost.
            * Prediction should be numpy.ndarray. Shape (1, m).
        """
        A1, output = self.forward_prop(X)
        cost = self.cost(Y, output)
        output = np.where(output >= 0.5, 1, 0)
        return output.astype(int), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
        * @X: numpy.ndarray. Shape: (nx, m). Contains input data.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        * @A1: output of the hidden layer
        * @A2: predicted output
        * @alpha: learning rate
        Updates private attributes __W1, __W2, __b1, __b2
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1/m)*(np.matmul(dz2, A1.T))
        db2 = (1/m)*(np.sum(dz2, axis=1, keepdims=True))
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

        dz1 = (np.matmul(self.__W2.T, dz2))*(A1*(1-A1))
        dw1 = (1/m)*(np.matmul(dz1, X.T))
        db1 = (1/m)*(np.sum(dz1, axis=1, keepdims=True))
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
