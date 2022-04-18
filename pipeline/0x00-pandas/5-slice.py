#!/usr/bin/env python3
"""Script that slices a pd.DataFrame along the columns 'High', 'Low',
'Close', and 'Volume_BTC', taking every 60th row."""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == '__main__':
    file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    df = from_file(file_name, ',')
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]

    print(df.tail())
