#!/usr/bin/env python3
"""Contains the class BayesianOptimization"""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """Args:
            f: function to optimize
            X_init (ndarray)(t,1): represents inputs already sampled
            Y_init (ndarray)(t,1): represents outputs for each input of X_init
            t (int): no. initial samples
            bounds (tuple)(min,max): bounds of the space in which to look for
                optimal point
            ac_samples: no. samples that should be analyzed during acquisition
            l: length parameter for the kernel
            sigma_f (float): std. dev.
            xsi: exploration-exploitation factor for acquisition
            minimize (bool): determines whether optimization should be
                performed for minimization (True) or maximization (False)

        Public instance attributes:
            f: black-box function
            gp: instance of the class GaussianProcess
            X_s: acquisition sample points, evenly spaced between min and max
            xsi: exploration-exploitation factor
            minimize: boolean, determines minimization or maximization
        """
        min, max = bounds

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(min, max, ac_samples)[:, np.newaxis]
        self.xsi = xsi
        self.minimize = minimize
