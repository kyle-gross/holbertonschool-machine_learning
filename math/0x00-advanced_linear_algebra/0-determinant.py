#!/usr/bin/env python3
"""Contains the function determinant()"""

import numpy as np


def determinant(matrix):
    """Calculates the determinant of a matrix

    Arg:
        matrix (list): list of lists whose determinant should be calculated

    Returns:
        determinant of matrix
    """
    if matrix == [[]]:
        return 1

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for i in matrix:
        if type(i) is not list:
            raise TypeError('matrix must be a list of lists')

    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    return int(np.linalg.det(matrix))
