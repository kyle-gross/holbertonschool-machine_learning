#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

if __name__ == '__main__':
    file_name1 = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    file_name2 = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'

    df1 = from_file(file_name1, ',')
    df2 = from_file(file_name2, ',')

    df1 = df1.loc[
        (df1['Timestamp'] >= 1417411980) &
        (df1['Timestamp'] <= 1417417980)
    ]
    df2 = df2.loc[
        (df2['Timestamp'] >= 1417411980) &
        (df2['Timestamp'] <= 1417417980)
    ]
    df1.set_index('Timestamp', inplace=True)
    df2.set_index('Timestamp', inplace=True)

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.reorder_levels([1, 0], axis=0)
    df.sort_index(inplace=True)

    print(df)
