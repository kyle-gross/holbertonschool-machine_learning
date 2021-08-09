#!/usr/bin/env python3
"""
Contains the class DeepNeuralNetwork.
"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork():
    """
    Defines a deep neural network performing binary classification.
    """
    def __init__(self, nx, layers):
        """
        * @nx: number of input features
            * Must be int. Must be > 0.
        * @layers: list representing # of nodes in each layer of network.
            * Must be list. Must NOT be empty.
            * 1st value represents the number of nodes in the first layer.
            * Elements must ALL be positive integers.
        Sets the public instance attributes:
            * L: # of layers in neural network.
            * cache: dict to hold all intermediary values of network.
            * weights: dict to hold weights and biases of network.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        else:
            self.__L = len(layers)
            self.__cache = {}
            self.__weights = {}
            prev = nx
            for i in range(len(layers)):
                if type(layers[i]) is not int or layers[i] < 1:
                    raise TypeError('layers must be a list of positive integers')
                w = np.random.randn(layers[i], prev) * np.sqrt(2/prev)
                prev = layers[i]
                self.__weights['W{}'.format(i + 1)] = w
                dim = len(self.__weights['W{}'.format(i + 1)])
                self.__weights['b{}'.format(i + 1)] = np.zeros((dim, 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network.
        * @X: numpy.ndarray. Shape: (nx, m). Contains input data.
        * Updates private attribute __cache
            * Activated outputs of each layer should be saved to cache
              dict.
        Return: output of neural network and cache
        """
        self.__cache['A0'] = X
        x = X
        for i in range(self.__L):
            W = self.__weights['W{}'.format(i+1)]
            b = self.__weights['b{}'.format(i+1)]
            z = np.matmul(W, x) + b
            A = 1/(1 + np.exp(-z))
            x = A
            self.__cache['A{}'.format(i+1)] = A
        return self.__cache['A{}'.format(self.__L)], self.__cache

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
        output, A = self.forward_prop(X)
        cost = self.cost(Y, output)
        output = np.where(output >= 0.5, 1, 0)
        return output.astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        * @cache: dict containing intermediary values of the network.
        * @alpha: learning rate.
        * Updates private attribute: __weights.
        """
        m = Y.shape[1]
        back_prop = {}
        for i in range(self.L, 0, -1):
            A_next = cache['A{}'.format(i-1)]
            if i == self.L:
                back_prop['dz{}'.format(i)] = (cache['A{}'.format(i)] - Y)
            else:
                A = cache['A{}'.format(i)]
                dz_prev = back_prop['dz{}'.format(i+1)]
                back_prop['dz{}'.format(i)] = \
                    (np.matmul(W.T, dz_prev)*(A*(1-A)))
            dz = back_prop['dz{}'.format(i)]
            dW = (1/m)*(np.matmul(dz, A_next.T))
            db = (1/m)*np.sum(dz, axis=1, keepdims=True)
            W = self.weights['W{}'.format(i)]
            self.__weights['W{}'.format(i)] = W - (alpha * dW)
            self.__weights['b{}'.format(i)] = \
                self.weights['b{}'.format(i)] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network.
        * @X: numpy.ndarray. Shape: (nx, m). Contains input data.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        * @iterations: number of iterations to train over.
            * Must be int. Must be > 0.
        * @alpha: learning rate.
            * Must be float. Must be > 0.
        * @verbose: bool, if true displays training progress
        * @graph: bool, if true displays training accuracy graph
        * @step: int, # of iterations to display @verbose info
        * Updated private attributes: __weights and __cache
        Return: Evaluation of training after @iterations have occurred.
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
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
            x_axis = np.arange(0, iterations + 1, step)
        for i in range(iterations):
            AL, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 and verbose is True:
                cost = self.cost(Y, AL)
                print('Cost after {} iterations: {}'.format(i, cost))
            if i % step == 0 and graph is True:
                cost = self.cost(Y, AL)
                costs.append(cost)
        if verbose is True:
            cost = self.cost(Y, AL)
            print('Cost after {} iterations: {}'.format(iterations, cost))
        if graph is True:
            cost = self.cost(Y, AL)
            costs.append(cost)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.plot(x_axis, costs, 'b')
            plt.show()
        return self.evaluate(X, Y)
