#!/usr/bin/env python3
"""
Contains the function 'matrix_transpose' which returns the transpose
of a 2d matrix
"""
def matrix_transpose(matrix):
    """
    Returns the transpose of a 2d matrix
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
