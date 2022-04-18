#!/usr/bin/env python3
"""Script which fills in the missing data points in pd.DataFrame"""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == '__main__':
    file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    df = from_file(file_name, ',')

    df.drop(columns=['Weighted_Price'], inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    df['High'].fillna(value=df.Close, inplace=True)
    df['Low'].fillna(value=df.Close, inplace=True)
    df['Open'].fillna(value=df.Close, inplace=True)
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)

    print(df.head())
    print(df.tail())
