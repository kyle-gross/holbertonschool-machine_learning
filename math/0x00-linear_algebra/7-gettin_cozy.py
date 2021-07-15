#!/usr/bin/env python3
"""
Contains the function 'cat_matrices' which concatenates 2 matrices
along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates 2 matrices along a specific axis
    * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    * You can assume all elements in the same dimension are of the same
      type/shape
    * You must return a new matrix
    * If the two matrices cannot be concatenated, return None
    """
    new_mat = []

    if axis < 0:
        return None

    if axis == 0:
        [new_mat.append(row.copy()) for row in mat1]
        [new_mat.append(row.copy()) for row in mat2]

    if axis != 0:
        [new_mat.append(row.copy()) for row in mat1]
        for i in range(len(new_mat)):
            [new_mat[i].append(mat2[i][0])]

    return new_mat
