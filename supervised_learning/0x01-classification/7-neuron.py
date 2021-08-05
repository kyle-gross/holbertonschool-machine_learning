#!/usr/bin/env python3
"""
Contains the class Neuron.
"""
import matplotlib.pyplot as plt
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates the gradient descent on the neuron.
        * @X: numpy.ndarray with shape (nx, m). Contains input data.
        * @Y: numpy.ndarray with shape (1, m). Contains correct labels.
        * @A: numpy.ndarray with shape (1, m). Contains neuron output.
        * @alpha: learning rate
        Updates private attributes __W and __b
        """
        m = X.shape[1]
        dw = (1/m)*(np.matmul(X, (A-Y).T))
        db = (1/m)*(np.sum(A - Y))
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron.
        * @X: numpy.ndarray with shape (nx, m). Contains input data.
        * @Y: numpy.ndarray with shape (1, m). Contains correct labels.
        * @iterations: number of iterations to train over.
            * Must be int. Must be > 0.
        * @alpha: learning rate.
            * Must be float. Must be > 0.
        * @verbose: bool - defines whether or not to print information.
        * @graph: bool - defines whether or not to graph information.
        Return: evaluation of training data after @iterations of training
                have occurred.
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive number')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
            costs = []
            x_axis = []
        for i in range(0, iterations + 1):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if (i in (0, iterations) or i % step == 0)\
               and (verbose is True or graph is True):
                cost = self.cost(Y, self.__A)
                costs.append(cost)
                x_axis.append(i)
                if verbose is True:
                    print('Cost after {} iterations: {}'.format(i, cost))
        if graph is True:
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.plot(x_axis, costs, 'b')
            plt.show()
        return self.evaluate(X, Y)
