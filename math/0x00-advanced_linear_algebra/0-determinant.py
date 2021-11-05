#!/usr/bin/env python3
"""Contains the function determinant()"""


def determinant(matrix):
    """Calculates the determinant of a matrix

    Arg:
        matrix (list): list of lists whose determinant should be calculated

    Returns:
        determinant of matrix
    """
    if matrix == [[]]:
        return 1

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for i in matrix:
        if type(i) is not list:
            raise TypeError('matrix must be a list of lists')

    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    return det_helper(matrix)


def det_helper(A, total=0):
    """Recursive helper function for finding determinant"""
    indices = list(range(len(A)))

    if len(A) == 2 and len(A[0]) == 2:
        return A[0][0] * A[1][1] - A[1][0] * A[0][1]

    # For each focus column
    for fc in indices:
        # Find the submatrix
        As = copy_matrix(A)  # make a copy
        As = As[1:]  # remove first row
        height = len(As)

        # For each remaining row in submatrix
        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc+1:]  # zero focus column elements

        sign = (-1) ** (fc % 2)  # alternate signs for submatrix multiplier
        sub_det = det_helper(As)  # pass submatrix recursively
        total += sign * A[0][fc] * sub_det

    return total


def copy_matrix(matrix):
    """Creates and returns a copy of a matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    matrix_copy = []

    # Fill matrix_copy with zeros
    while len(matrix_copy) < rows:
        matrix_copy.append([])
        while len(matrix_copy[-1]) < cols:
            matrix_copy[-1].append(0.0)

    # Copy matrix values to matrix_copy
    for i in range(rows):
        for j in range(cols):
            matrix_copy[i][j] = matrix[i][j]

    return matrix_copy
