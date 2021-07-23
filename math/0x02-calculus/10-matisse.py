#!/usr/bin/env python3
"""
Contains the function 'poly_derivative' which calculates the derivative
of a polynomial.
    * poly is a list of coefficients representing a polynomial
        * the index of the list represents the power of x that
          the coefficient belongs to
        * Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    * If poly is not valid, return None
    * If the derivative is 0, return [0]
    * Return a new list of coefficients representing the
      derivative of the polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.
    """
    if len(poly) == 0 or type(poly) is not list:
        return None

    return [poly[i] * i for i in range(1, len(poly))]
