#!/usr/bin/env python3
"""
Contains the class DeepNeuralNetwork.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
                    raise TypeError(
                        'layers must be a list of positive integers')
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
        def sigmoid(act):
            """
            Sigmoid activation function.
            """
            return 1/(1 + np.exp(-act))

        def softmax(act):
            """
            Softmax activation function.
            """
            return np.exp(act)/np.sum(np.exp(act), axis=0)

        self.__cache['A0'] = X
        for i in range(self.L):
            W = self.__weights['W{}'.format(i+1)]
            x = self.__cache['A{}'.format(i)]
            b = self.__weights['b{}'.format(i+1)]
            z = np.matmul(W, x) + b
            if i == self.L - 1:
                act = softmax(z)
            else:
                act = sigmoid(z)
            self.__cache['A{}'.format(i+1)] = act
        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates cost of model using logistic regression.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        * @A: numpy.ndarray. Shape: (1, m). Contains activated output.
        Return: cost
        """
        m = A.shape[1]
        loss = np.sum(-Y * np.log(A))
        return loss / m

    def evaluate(self, X, Y):
        """
        Evaluates neural network's predictions.
        * @X: numpy.ndarray. Shape: (nx, m). Contains input data.
        * @Y: numpy.ndarray. Shape: (1, m). Contains correct labels.
        Return: Neuron's prediction and cost.
            * Prediction should be numpy.ndarray. Shape (1, m).
        """
        def one_hot_encode(Y, classes):
            """
            Converts a numeric label vector into a one-hot matrix.
            """
            if type(Y) is not np.ndarray or type(classes) is not int:
                return None
            if classes < 2 or classes < np.max(Y):
                return None
            onehot_encode = np.zeros((classes, Y.size))
            onehot_encode[Y, np.arange(Y.size)] = 1
            return onehot_encode

        def one_hot_decode(one_hot):
            """
            Converts a one-hot matrix into a vector of labels.
            """
            if type(one_hot) is not np.ndarray:
                return None
            if one_hot.ndim != 2:
                return None
            return np.argmax(one_hot, axis=0)

        output, A = self.forward_prop(X)
        cost = self.cost(Y, output)
        classes = Y.shape[0]
        decoded = one_hot_decode(output)
        encode = one_hot_encode(decoded, classes)
        return encode.astype(int), cost

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

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        * @filename: the file to which the object should be saved.
            * Add '.pkl' extension if @filename does not already have it.
        """
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        * @filename: file from which the object should be loaded.
        Return: loaded object, or None if @filename doesn't exist.
        """
        try:
            with open(filename, 'rb') as input_file:
                obj = pickle.load(input_file)
            return obj
        except Exception:
            return None
