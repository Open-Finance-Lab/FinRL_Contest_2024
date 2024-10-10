import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from task2_stocks import get_stock_data
from task2_news import download_news
from task2_sentiment import generate_sentiment


API_KEY = "gja_IxOHjkAnbpn4mQ0qax4CZcIpZYXk"  # TODO provide a free tier polygon api key to run the baseline. You are free to use other news data source, you may also provide your news

# Date ranges for the starter solution
END_DATE = "2024-10-01"
START_DATE = "2024-09-20"

"""a very simple env whost state space is only the data"""
STOCK_TICKERS_HIGHEST_CAP_US = [
    "AAPL",
    # "NVDA",
    # "GOOG",
    # "AMZN",
    # "META",
    # "BRK-B",
    # "AVGO",
    # "LLY",
    # "TSLA",
    # "WMT",
]


"""load data and make dset - first we load in the ticker data for each ticker, then we enrich that with news data"""
stock_data = get_stock_data(STOCK_TICKERS_HIGHEST_CAP_US, START_DATE, END_DATE)

"""load model and env"""
from task2_env import Task2Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
# model = torch.nn.DataParallel(model)
# model.to(device)

task2env = Task2Env(
    model,
    tokenizer,
    STOCK_TICKERS_HIGHEST_CAP_US,
    stock_data,
    (-2, 2),
    max_steps=stock_data.shape[0]
    - 4,  # drop 4 for the future price calculation step and the count at 0 offset.
)


state = task2env.reset()
actions = []
rewards = []

for _ in range(len(stock_data)):
    date, prices = state
    ticker_actions = {}
    done = False
    for t in prices.Ticker:
        news = download_news(
            t, date._date_repr, (date - timedelta(days=10))._date_repr, API_KEY
        )
        sentiment_score = generate_sentiment(
            tokenizer,
            model,
            device,
            news,
            prices.copy().drop("future_close", axis=1)[
                prices["Ticker"] == t
            ],  # drop future close from the state to prevent data leakage
        )
        ticker_actions[t] = sentiment_score

    state, reward, done, _ = task2env.step(ticker_actions)

    actions.append(ticker_actions)
    rewards.append(reward)

    if done:
        break


plt.figure(figsize=(10, 6))
plt.plot(rewards, marker="o", linestyle="-", color="b", label="Rewards")
plt.title("Rewards Over Time")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("test_rewards.png")
