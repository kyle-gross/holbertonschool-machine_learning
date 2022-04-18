#!/usr/bin/env python3
"""Script which renames a column and converts
timestamp values to datetime values"""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == '__main__':
    file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    df = from_file(file_name, ',')
    df = df[['Timestamp', 'Close']].rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    print(df.tail())
