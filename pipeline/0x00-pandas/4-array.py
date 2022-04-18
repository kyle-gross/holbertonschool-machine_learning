#!/usr/bin/env python3
"""Script that converts a pd.DataFrame into an np.array"""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == '__main__':
    file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    df = from_file(file_name, ',')

    A = df[['High', 'Close']].tail(10).to_numpy()

    print(A)
