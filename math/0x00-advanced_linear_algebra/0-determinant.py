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

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0

    for row in range(len(matrix)):
        matrix_copy = [col.copy() for col in matrix]  # Copy matrix
        matrix_copy.pop(0)  # Remove first row
        for col in range(len(matrix_copy)):
            matrix_copy[col].pop(row)  # Remove rows after using
        # Alternate signs for submatrix multiplier
        if row % 2 == 0:
            det += matrix[0][row] * determinant(matrix_copy)
        if row % 2 == 1:
            det -= matrix[0][row] * determinant(matrix_copy)

    return det
