import pandas as pd


def _get_news(ticker: str, start_date: str, end_date: str, dset: str) -> str:
    """Retrieve news content for a specific stock ticker within a date range.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The starting date for the news.
        end_date (str): The ending date for the news.
        dset (str): Path to the CSV dataset containing news.

    Returns:
        str: Concatenated news content including article titles and Textrank summaries.
    """
    df = pd.read_csv(dset)

    # Filter data by the given ticker and date range
    df = df[df["Stock_symbol"] == ticker]
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Concatenate relevant news content
    news_content = ""
    for _, row in df.iterrows():
        title = row["Article_title"] if pd.notna(row["Article_title"]) else ""
        textrank_summary = row["Textrank_summary"] if pd.notna(row["Textrank_summary"]) else ""
        news_content += f"{title}, {textrank_summary} -- "

    return news_content


def get_news(ticker: str, start_date: str, end_date: str, dset="task2_dsets/test/task2_test_news.csv") -> str:
    """Wrapper around _get_news to simplify calling with default parameters.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The starting date for the news.
        end_date (str): The ending date for the news.
        dset (str): Path to the CSV dataset containing news. Defaults to "task2_dsets/test/task2_test_news.csv".

    Returns:
        str: Aggregated news content for the given ticker within the date range.
    """
    return _get_news(ticker, start_date, end_date, dset)
