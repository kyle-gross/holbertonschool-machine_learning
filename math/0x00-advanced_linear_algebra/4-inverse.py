#!/usr/bin/env python3
"""Contains the function inverse()"""


def inverse(matrix):
    """Calculates the inverse of a matrix

    Args:
        matrix: list of list whose inverse should be calculated

    Returns:
        inverse of the matrix, or None if matrix is singular
    """
    inv_matrix = adjugate(matrix)
    det = determinant(matrix)

    if det == 0:
        return None

    det = 1 / det

    for row in range(len(inv_matrix)):
        for col in range(len(inv_matrix)):
            inv_matrix[row][col] *= det

    return inv_matrix


def adjugate(matrix):
    """Calculates the adjugate of a matrix

    Args:
        matrix: list os list whose adjugate matrix should be calculated

    Returns:
        Adjugate of the matrix
    """
    temp = cofactor(matrix)
    adj_matrix = [[] for row in temp]

    for row in range(len(temp)):
        for col in range(len(temp)):
            adj_matrix[col].append(temp[col][row])

    return adj_matrix


def cofactor(matrix):
    """Calculates the cofactor of a matrix

    Arg:
        matrix: list of list whose cofactors should be calculated

    Returns:
        cofactor of the matrix
    """
    co_matrix = minor(matrix)

    for row in range(len(co_matrix)):
        for col in range(len(co_matrix)):
            co_matrix[row][col] *= pow(-1, row+col)

    return co_matrix


def minor(matrix):
    """Calculates the minor of a matrix of a matrix

    Args:
        matrix: list of lists whose minor should be calculated

    Returns:
        minor of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    for row in range(len(matrix)):
        minor_row = []
        for col in range(len(matrix)):
            sub_matrix = [row.copy() for row in matrix]
            sub_matrix.pop(row)  # Remove current row index
            for i in range(len(sub_matrix)):
                sub_matrix[i].pop(col)  # Remove current col index
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix


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
