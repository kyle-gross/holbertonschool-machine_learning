#!/usr/bin/env python3
"""Contains the class GaussianProcess"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Sets public instance attributes X, Y, l, and sigma_f"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, Y_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices using
        the Radial Basis Function (RBF)

        Args:
            X1 (ndarray): shape (m,1)
            X2 (ndarray): shape (n,1)

        Returns:
            K (ndarray)(m,n): covariance kernel matrix
        """
        l = self.l
        K = np.exp(-((X1-X2.T)**2) / (2*(l**2)))
        return K * (self.sigma_f**2)
