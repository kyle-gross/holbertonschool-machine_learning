#!/usr/bin/env python3
"""
Contains the function 'matrix_shape' which calculates the shape
of a matrix.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.
    * You can assume all elements in the same dimension are of the same
      type/shape
    * The shape should be returned as a list of integers
    """

    shape = []

    shape.append(len(matrix))
    shape.append(len(matrix[0]))

    if type(matrix[0][0]) is list:
        shape.append(len(matrix[0][0]))

    return shape
