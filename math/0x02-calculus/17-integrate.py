#!/usr/bin/env python3
"""
Contains the function 'poly_integral' which calculates the
integral of a polynomial.
    * poly is a list of coefficients representing a polynomial
        * the index of the list represents the power of x
          that the coefficient belongs to
        * Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    * C is an integer representing the integration constant
    * If a coefficient is a whole number, it should be
      represented as an integer
    * If poly or C are not valid, return None
    * Return a new list of coefficients representing the
      integral of the polynomial
    * The returned list should be as small as possible
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    """
    if type(poly) is not list or type(C) is not int:
        return None

    if len(poly) == 0:
        return None

    integrals = [C]

    def int_check(x): return int(x) if x.is_integer() else x

    [integrals.append(int_check(poly[i] / (i + 1))) for i in range(len(poly))
     if type(poly[i]) in (int, float)]

    return integrals
