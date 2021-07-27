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
