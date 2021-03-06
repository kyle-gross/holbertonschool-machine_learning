#!/usr/bin/env python3
"""
Contains the Poisson class which represents a poisson distribution
"""
E = 2.7182818285


class Poisson():
    """
    Represents a poisson distribution
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
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        * Calculates the value of the PMF for a given number of “successes”
        * <k> is the number of “successes”
            * If <k> is not an integer, convert it to an integer
            * If <k> is out of range, return 0
        * Returns the PMF value for k
        The Poisson Distribution pmf is: P(x; μ) = (e^-μ * μ^x) / x!
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        μ = self.lambtha
        for i in range(k):
            factorial *= (i + 1)
        return (E ** -μ) * (μ ** k) / factorial

    def cdf(self, k):
        """
        *  Calculates the value of the CDF for a given number of “successes”
        * <k> is the number of “successes”
            * If <k> is not an integer, convert it to an integer
            * If <k> is out of range, return 0
        * Returns the CDF value for k
        Poisson cmf formula: F(x; λ) = ∑xi=0 ((e^-λ * λ^i) / i!)
            or
        Summation of pmf's within range 0 - k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
