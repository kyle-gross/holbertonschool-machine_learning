#!/usr/bin/env python3
"""Contains the class GaussianProcess"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Sets public instance attributes X, Y, l, sigma_f, and K"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices using
        the Radial Basis Function (RBF)

        Args:
            X1 (ndarray): shape (m,1)
            X2 (ndarray): shape (n,1)

        Returns:
            K (ndarray)(m,n): covariance kernel matrix
        """
        K = np.exp(-((X1 - X2.T)**2) / (2 * (self.l**2)))
        K *= (self.sigma_f**2)
        return K

    def predict(self, X_s):
        """Predicts the mean and std dev of points in a GP

        Args:
            X_s (ndarray)(s,1): contains points whose mean and std dev should
              be calculated.
                s: no. sample points

        Returns:
            mu, sigma
            mu (ndarray)(s,): contains mean for each point in X_s
            sigma (ndarray)(s,): contains the variance for each point in X_s
        """
        s = X_s.shape[0]
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        mu = K_ss.T.dot(np.linalg.inv(self.K)).dot(self.Y)
        sigma = K_ss - K_s.T.dot(np.linalg.inv(self.K)).dot(K_s)

        return mu.reshape(s,), np.diag(sigma)

    def update(self, X_new, Y_new):
        """Updates a GP

        Args:
            X_new (ndarray)(1,): new sample point
            Y_new (ndarray)(1,): new sample function value

        Updates attributes X, Y, and K
        """
        self.X = np.concatenate((self.X, X_new[:, None]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[:, None]), axis=0)
        self.K = self.kernel(self.X, self.X)
