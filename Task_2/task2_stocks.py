"""
You may modify the stock fetching and feature engineering pipeline.

We will import the downlaod_and_save_ohlcv_data function from this file and use that for testing. Should this file not work, we will use our function.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def _get_stock_data(tickers: list, start_date: str, end_date: str, filename: str = None):
    """Download OHLCV data for a list of stock tickers and save it to a CSV file."""
    all_data = pd.DataFrame()

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        stock_data["Ticker"] = ticker

        all_data = pd.concat([all_data, stock_data])
        all_data

    if filename:
        all_data.to_csv(filename)
        print(f"Data saved to {filename}")
    return all_data  # .reset_index()

def get_stock_data(tickers: list, start_date: str, end_date: str, filename: str = None):
    return _get_stock_data(tickers, start_date, end_date, filename)
