#!/usr/bin/env python3
"""Script which alters the pd.DataFrame such that the rows & columns are
transposed and the data is sorted in reverse chronological order"""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == '__main__':
    file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    df = from_file(file_name, ',')
    df = df[::-1].transpose()

    print(df.tail(8))
