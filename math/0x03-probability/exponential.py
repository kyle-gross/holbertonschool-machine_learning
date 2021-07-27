#!/usr/bin/env python3
"""
Contains the Exponential class which represents an exponential distribution
"""
E = 2.7182818285


class Exponential():
    """
    Represents an exponential distribution
    """
    def __init__(self, data=None, lambtha=1):
        """
        * <data> is a list of the data to be used to estimate the distribution
        * <lambtha> is the expected number of occurences in a given time frame
            * Saves <lambtha> as a float
        * If <data> is not given
            * Use the given lambtha
            * If lambtha is not a positive value or equals to 0, raise a
              ValueError: 'lambtha must be a positive value'
        * If <data> is given
            * Calculate the lambtha of data
            * If data is not a list, raise a TypeError
              with the message 'data must be a list'
            * If data does not contain at least two data points, raise a
                  ValueError: 'data must contain multiple values'
        """
        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """
        * Calculates the value of the PDF for a given time period
        * <x> is the time period
        * Returns the PDF value for <x>
        * If <x> is out of range, return 0
        Formula: f(x;λ) = λe^(-λx)
        """
        if x < 0:
            return 0
        λ = self.lambtha
        return λ * E ** (-λ * x)

    def cdf(self, x):
        """
        * Calculates the value of the CDF for a given time period
        * <x> is the time period
        * Returns the PDF value for <x>
        * If <x> is out of range, return 0
        Formula: f(x;λ) = 1 - e^(-λx)
        """
        if x < 0:
            return 0
        λ = self.lambtha
        return 1 - E ** -(λ ** x)
