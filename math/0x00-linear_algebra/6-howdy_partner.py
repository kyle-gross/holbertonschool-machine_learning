#!/usr/bin/env python3
"""
Contains the function 'cat_arrays' which concatenates 2 arrays
"""


def cat_arrays(arr1, arr2):
    """
    Concatenates 2 arrays
    * You can assume that arr1 and arr2 are lists of ints/floats
    * You must return a new list
    """
    new_arr = []

    [new_arr.append(i) for i in arr1]
    [new_arr.append(i) for i in arr2]

    return new_arr
