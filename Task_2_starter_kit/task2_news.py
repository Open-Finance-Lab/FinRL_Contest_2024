"""
Do not change this. We will be using this setup for testing your submissions.
"""

import pandas as pd


def _get_news(ticker: str, start_date: str, end_date: str, dset: str):

    # Read in csv file
    df = pd.read_csv(dset)
    # filter by ticker and date range
    df = df[df["Stock_symbol"] == ticker]

    df = df[
        (df["Date"] >= start_date) & (df["Date"] <= end_date)
    ]  # in default imp this is a five day range

    news_content = ""
    for _, row in df.iterrows():
        title = row["Article_title"] if pd.notna(row["Article_title"]) else ""
        textrank_summary = (
            row["Textrank_summary"] if pd.notna(row["Textrank_summary"]) else ""
        )
        news_content += f"{title}, {textrank_summary} -- "

    return news_content


def get_news(
    ticker: str, start_date: str, end_date: str, dset="task2_news.csv"
):
    return _get_news(ticker, start_date, end_date, dset)
