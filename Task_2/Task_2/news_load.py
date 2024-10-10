import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# Your Polygon.io API key
API_KEY = "your_polygon_io_api_key"


# Function to fetch OHLCV data for a specific ticker
def download_news(ticker: str, start_date: str, end_date: str, api_key=None):

    news_url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.lt={start_date}&published_utc.gt={end_date}&limit=10&apiKey={api_key}"

    news_data = requests.get(news_url).json()

    """extract results object"""
    results = news_data["results"]

    """data['results'][0]['title'] is where we can access the titles"""
    headlines = ""
    for article in results:
        headlines += f"{article['title']}, {article.get('description', '')} -- "

    return headlines
