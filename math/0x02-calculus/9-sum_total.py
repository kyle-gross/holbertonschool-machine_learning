#!/usr/bin/env python3
"""
This module contains the function 'summaation_i_squared' which
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

    if n == 1:
        # Base case - stop when n reaches 1
        return n ** 2 

    return summation_i_squared(n - 1) + n ** 2
