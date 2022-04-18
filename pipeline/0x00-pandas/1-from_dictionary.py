#!/usr/bin/env python3
"""Creates a pd.DataFrame from a python dictionary"""

import pandas as pd


if __name__ == '__main__':
    dict = {
        'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']
    }
    rows = ['A', 'B', 'C', 'D']

    df = pd.DataFrame(dict, index=rows)

    print(df)
