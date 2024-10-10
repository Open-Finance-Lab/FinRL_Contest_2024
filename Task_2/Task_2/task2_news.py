"""
Do not change this. We will be using this setup for testing your submissions.
"""

import requests


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
