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
    [new_mat.append(row.copy()) for row in mat1]

    if axis < 0:
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        [new_mat.append(row.copy()) for row in mat2]
        return new_mat

    if len(mat1) != len(mat2):
        return None

    for i in range(len(new_mat)):
        for j in mat2[i].copy():
            new_mat[i].append(j)

    return new_mat
