#!/usr/bin/env python3
"""
Contains the function 'add_arrays' which returns the addition of
two arrays
    * You can assume that arr1 and arr2 are lists of ints/floats
    * You must return a new list
    * If arr1 and arr2 are not the same shape, return None
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays
    """
    new_arr = []

    if len(arr1) != len(arr2):
        return None

    for i in range(len(arr1)):
        new_arr.append(arr1[i] + arr2[i])

    return new_arr
