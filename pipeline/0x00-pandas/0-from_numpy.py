#!/usr/bin/env python3
"""Contains the function from_numpy()"""

import numpy as np
import pandas as pd
import string


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray"""
    col_headers = list(string.ascii_uppercase)
    col_len = array.shape[1]
    df = pd.DataFrame(array, columns=col_headers[:col_len])

    return df


if __name__ == '__main__':
    np.random.seed(0)
    A = np.random.randn(5, 8)
    print(from_numpy(A))
    B = np.random.randn(9, 3)
    print(from_numpy(B))
