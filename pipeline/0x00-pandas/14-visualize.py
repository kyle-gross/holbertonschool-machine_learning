#!/usr/bin/env python3
"""Script which visualizes the pd.DataFrame
* Plot data from 2017 and beyond at daily intervals
* Remove column Weighted_Price
* Convert timestamp values to date values
* Index the data frame on Date
* Missing values in High, Low, Open, and Close should be set to previous
  row's Close value
* Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

if __name__ == '__main__':
    file_name = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    df = from_file(file_name, ',')

    # Drop unnecessary column and convert date to datetime
    df.drop(columns='Weighted_Price', inplace=True)
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df.set_index('Date', inplace=True)
    df = df[df.index.year >= 2017]

    # Fill in for missing values
    df['Close'].fillna(method='ffill', inplace=True)
    df['High'].fillna(df.Close, inplace=True)
    df['Low'].fillna(df.Close, inplace=True)
    df['Open'].fillna(df.Close, inplace=True)
    df['Volume_(BTC)'].fillna(0, inplace=True)
    df['Volume_(Currency)'].fillna(0, inplace=True)

    df_plot = pd.DataFrame()
    df_plot['High'] = df['High'].resample('D').max()
    df_plot['Low'] = df['Low'].resample('D').min()
    df_plot['Open'] = df['Open'].resample('D').mean()
    df_plot['Close'] = df['Close'].resample('D').mean()
    df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
    df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

    df_plot.plot()
    plt.show()
