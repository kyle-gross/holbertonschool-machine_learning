#!/usr/bin/env python3
"""
Contains the function 'add_matrices2D' which adds two matrices
element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds 2 matrices element-wise
    * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same
      type/shape
    * You must return a new matrix
    * If mat1 and mat2 are not the same shape, return None
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat2))]


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.
    * You can assume all elements in the same dimension are of the same
      type/shape
    * The shape should be returned as a list of integers
    """
    shape = []

    if type(matrix) is list:
        shape.append(len(matrix))
        shape.extend(matrix_shape(matrix[0]))

    return shape
