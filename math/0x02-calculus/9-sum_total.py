#!/usr/bin/env python3
"""
This module contains the function 'summation_i_squared' which
calculates sum_{i=1}^{n} i^2
    * <n> is the stopping condition
    * Return the integer value of the sum
    * If <n> is not a valid number, return None
    * You are not allowed to use any loops
"""


def summation_i_squared(n):
    """
    Calculates sum_{i=1}^{n} i^2
    """
    if n < 1:
        # Return None if <n> is not a valid number
        return None

    # Mathematical formula for Python sum of series
    # 1^2 + 2^2 + 3^2...
    # = (n(n+1)(2n+1)) / 6
    return int((n * (n + 1) * (2 * n + 1)) / 6)
