#!/usr/bin/env python3
"""
Contains the Normal class which represents a normal distribution
"""
PI = 3.1415926536
E = 2.7182818285


class Normal():
    """
    Represents a normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        * <data> is a list of the data to be used to estimate the distribution
        * <mean> is the mean of the distribution
        * <stddev> is the standard deviation of the distribution
        * Sets the instance attributes <mean> and <stddev>
            * Saves <mean> and <stddev> as floats
        * If <data> is not given
            * Use the given <mean> and <stddev>
            * If <stddev> > 0, raise a ValueError:
              'stddev must be a positive value'
        * If <data> is given:
            * Calculate the mean and standard deviation of <data>
            * If <data> is not a list, raise a TypeError:
              'data must be a list'
            * If <data> does not contain at least two data points, ValueError:
            'data must contain multiple values'
        """
        if data is None:
            if stddev > 0:
                self.mean = float(mean)
                self.stddev = float(stddev)
            else:
                raise ValueError('stddev must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                n = len(data)
                self.mean = float(sum(data) / n)
                deviations = [(x - self.mean) ** 2 for x in data]
                variance = sum(deviations) / n
                self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        * Calculates the z-score of a given x-value
        * <x> is the x-value
        * Returns the z-score of <x>
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        * Calculates the x-value of a given z-score
        * <z> is the z-score
        * Returns the x-value of <z>
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        * Calculates the value of the PDF for a given x-value
        * <x> is the x-value
        * Returns the PDF value for <x>
        Formula: 1/σ(2π^0.5)*e^-0.5*(x-μ/σ)^2
        """
        μ = self.mean
        σ = self.stddev
        exponent = -0.5 * ((x - μ) / σ) ** 2
        return 1 / (σ * (2 * PI) ** 0.5) * (E ** exponent)

    def cdf(self, x):
        """
        * Calculates the value of the CDF for a given x-value
        * <x> is the x-value
        * Returns the CDF value for <x>
        Formula: 0.5*[1+erf(x-μ/σ*2**0.5)]
        """
        μ = self.mean
        σ = self.stddev
        x = (x - μ) / (σ * 2 ** 0.5)
        erf = (2 / PI ** 0.5) * (x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42
                                 + x ** 9 / 216)
        return 0.5 * (1 + erf)
