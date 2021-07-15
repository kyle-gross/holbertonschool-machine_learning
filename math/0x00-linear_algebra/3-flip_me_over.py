#!/usr/bin/env python3
"""
Contains the function 'matrix_transpose' which returns the transpose
of a 2d matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2d matrix
    """
    x = matrix

    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
