import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

from task2_news import get_news
from task2_signal import generate_eval_signal
from task2_config import Task2Config


# Date ranges for the starter solution. You may withold some of the training data and use it as validation data
END_DATE = None
START_DATE = None

"""a very simple env whost state space is only the data"""
STOCK_TICKERS_HIGHEST_CAP_US = [
    "AAPL",
    "NVDA",
    "GOOG",
    "AMZN",
    "MSFT",
    "XOM",
    "WMT",
]

eval_config = Task2Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    bnb_config=BitsAndBytesConfig(load_in_8bit=True),
    tickers=STOCK_TICKERS_HIGHEST_CAP_US,
    end_date=END_DATE,
    start_date=START_DATE,
    lookahead=3,
    signal_strengh=10,
    num_short=3,
    num_long=3,
)

"""load model and env"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # map to auto for multi gpu


tokenizer = AutoTokenizer.from_pretrained(eval_config.model_name)
model = AutoModelForCausalLM.from_pretrained(
    eval_config.model_name,
    quantization_config=eval_config.bnb_config,
    device_map="auto",
)


"""Implement evaluation strategy: The evaluation strategy starts with N in starting cash. At each trading step we generate signals for each stock."""
trade_win_loss = []  # 1 for win 0 for loss
daily_returns = []

# data

stock_data = pd.read_csv("task2_eval_stocks.csv")

# this is simply for logging to see how your model is performing. We will not be evaluating on threshold trading returns
logging_cum_returns_df_threshold_based = []

# keeping track of cumulative returns
eval_cum_returns_data = []

for date in tqdm(eval_config.eval_dates, desc="Evaluating..."):
    prices = stock_data[stock_data["Date"] == date._date_repr]

    if prices.empty:
        print(f"No data found for day {date._date_repr}")
        continue

    # each day for each ticker get sentiment and initiate a trade
    ticker_signals = {}

    for ticker in eval_config.tickers:
        news = get_news(
            ticker,
            (
                date - timedelta(days=1)
            )._date_repr,  # get news from the previous day to prevent post market close data leakage
            (date - timedelta(days=11))._date_repr,
            "task2_news.csv",  # you can change this to the eval news set that you create to test your model
        )

        signal_score = generate_eval_signal(
            tokenizer,
            model,
            device,
            news,
            prices.copy().drop("future_close", axis=1)[prices["Ticker"] == ticker],
            eval_config.signal_strengh,
            eval_config.threshold,
        )

        ticker_signals[ticker] = signal_score

        close_price = prices.loc[prices["Ticker"] == ticker, "Close"].item()  # buy at today's close
        future_price = prices.loc[prices["Ticker"] == ticker, "future_close"].item()  # sell at future close

        ### THRESHOLD BASED TRADING FOR INFO ON MODEL BEHAVIOR ###
        # using generated signal initiate a trade
        if signal_score >= eval_config.threshold:
            # long, sell at c price
            value_change = (future_price - close_price) / close_price
        elif signal_score <= -1 * eval_config.threshold:
            # short, sell at c price and buy back at f price
            value_change = -1 * ((future_price - close_price) / close_price)  # good if negative % change
        else:
            value_change = 0

        logging_cum_returns_df_threshold_based.append({"Date": date, "Ticker": ticker, "ValueChange": value_change})

    ### EVAL STRATEGY ###
    sorted_tickers = sorted(ticker_signals.items(), key=lambda x: x[1], reverse=True)
    sorted_thresh_tickers_long = [ticker for ticker in sorted_tickers if ticker[1] > eval_config.threshold]
    sorted_thresh_tickers_short = [ticker for ticker in sorted_tickers if ticker[1] < -eval_config.threshold]
    long_tickers = sorted_thresh_tickers_long[: eval_config.num_long]  # Highest signal scores
    short_tickers = sorted_thresh_tickers_short[eval_config.num_short :]  # Lowest signal scores

    eval_rets = []
    for ticker, signal_score in long_tickers + short_tickers:
        close_price = prices.loc[prices["Ticker"] == ticker, "Close"].item()
        future_price = prices.loc[prices["Ticker"] == ticker, "future_close"].item()

        # Calculate value change based on long or short position
        if ticker in dict(long_tickers):
            value_change = (future_price - close_price) / close_price  # Long trade
        else:
            value_change = -1 * ((future_price - close_price) / close_price)  # Short trade

        eval_rets.append(value_change)

    # Track the trade result
    eval_cum_returns_data.append({"Date": date, "MeanEvalReturn": np.mean(eval_rets), "win_loss": value_change >= 0})


### LOGGING - FOR YOUR INFORMATION, YOU MAY CHANGE THIS SECTION TO ADD ADDITIONAL LOGGING ###
# Create DataFrames
logging_cum_returns_df_threshold_based = pd.DataFrame(logging_cum_returns_df_threshold_based)
logging_cum_returns_df_threshold_based["CumulativeReturn"] = logging_cum_returns_df_threshold_based.groupby("Ticker")[
    "ValueChange"
].cumsum()

# calculate cumulative returns and win loss
eval_cum_returns_df = pd.DataFrame(eval_cum_returns_data)
eval_cum_returns_df["cumulative_return"] = (1 + eval_cum_returns_df["MeanEvalReturn"]).cumprod() - 1
win_rate = eval_cum_returns_df["win_loss"].mean()  # Mean gives the proportion of wins
loss_rate = 1 - win_rate  # Loss rate is complementary to win rate

# Save DataFrames to CSV
logging_cum_returns_df_threshold_based.to_csv("task2_threshold_based_cumulative_returns.csv", index=False)
eval_cum_returns_df.to_csv("task2_eval_returns.csv", index=False)

# Plot cumulative returns for each ticker
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Log Cumulative Returns per Ticker
for ticker in STOCK_TICKERS_HIGHEST_CAP_US:
    ticker_data = logging_cum_returns_df_threshold_based[logging_cum_returns_df_threshold_based["Ticker"] == ticker]
    axes[0].plot(ticker_data["Date"], np.log1p(ticker_data["CumulativeReturn"]), label=ticker)

axes[0].set_xlabel("Date")
axes[0].set_ylabel("Cumulative Return")
axes[0].set_title("Cumulative and Absolute Returns by Ticker")
axes[0].legend(title="Ticker")
axes[0].grid(True)

# Plot 2: Mean Evaluation Cumulative Returns
axes[1].plot(eval_cum_returns_df["Date"], eval_cum_returns_df["MeanEvalReturn"], color="blue", marker="o")
axes[1].plot(eval_cum_returns_df["Date"], eval_cum_returns_df["MeanEvalReturn"], color="red", marker="o")

axes[1].set_xlabel("Date")
axes[1].set_ylabel("Mean Eval Return")
axes[1].set_title("Mean Evaluation Cumulative Return of the Run")
axes[1].grid(True)

# Save the plot
plot_filename = "task2_cumulative_returns_plot.png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

# Show the plot
plt.show()

print("#######################")
print(
    f"Cumulative return: {eval_cum_returns_df['cumulative_return'][len(eval_cum_returns_df['cumulative_return'])-1]}, win rate: {win_rate}, loss rate: {loss_rate}"
)
print("#######################")
