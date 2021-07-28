#!/usr/bin/env python3
"""
Contains the Binomial class which represents a binomial distribution
"""


class Binomial():
    """
    Represents a binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        * <data> is a list of the data to be used to estimate the distribution
        * <n> is the number of Bernoulli trials
        * <p> is the probability of a “success”
            * Sets the instance attributes <n> and <p>
            * Saves <n> as an integer and <p> as a float
        * If <data> is not given
            * Use the given <n> and <p>
            * If <n> is not a positive value, raise a ValueError:
              'n must be a positive value'
            * If <p> is not a valid probability, raise a ValueError:
            'p must be greater than 0 and less than 1'
        * If data is given:
            * Calculate <n> and <p> from <data>
            * Round <n> to the nearest integer (rounded, not casting!)
            * If data is not a list, raise a TypeError:
              'data must be a list'
            * If data does not contain at least two data points, ValueError:
              'data must contain multiple values'
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            elif p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = n
                self.p = p
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = float(sum(data) / len(data))
                deviations = [(x - mean) ** 2 for x in data]
                variance = sum(deviations) / len(data)
                q = variance / mean
                p = 1 - q
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p

    def pmf(self, k):
        """
        * Calculates the value of the PMF for a given number of “successes”
        * <k> is the number of “successes”
            * If <k> is not an integer, convert it to an integer
            * If <k> is out of range, return 0
        * Returns the PMF value for <k>
        Formula: (n..k)*p^k*q^n-k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        n = self.n
        p = self.p

        def factorial(x):
            if x == 0:
                return 1
            return x if x <= 1 else x * factorial(x - 1)
        n_factorial = factorial(n)
        k_factorial = factorial(k)
        nk_factorial = factorial(n - k)
        nk = n_factorial / (k_factorial * nk_factorial)
        return nk * p ** k * (1 - p) ** (n - k)

    def cdf(self, k):
        """
        * Calculates the value of the CDF for a given number of “successes”
        * <k> is the number of “successes”
            * If <k> is not an integer, convert it to an integer
            * If <k> is out of range, return 0
        * Returns the CDF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
