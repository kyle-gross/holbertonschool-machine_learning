#!/usr/bin/env python3
"""
Contains the function 'mat_mul' which performs matrix multiplication
    * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same
      type/shape
    * You must return a new matrix
    * If the two matrices cannot be multiplied, return None
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None

    return [[sum(a * b for a, b in zip(mat1_row, mat2_col))
            for mat2_col in zip(*mat2)]
            for mat1_row in mat1]
