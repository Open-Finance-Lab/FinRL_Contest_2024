import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

from task2_news import get_news
from task2_signal import generate_eval_signal
from task2_config import Task2Config

# Global Variables
START_DATE = None
END_DATE = None

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

STOCK_TICKERS_HIGHEST_CAP_US = [
    "AAPL",
    "NVDA",
    "GOOG",
    "AMZN",
    "MSFT",
    "XOM",
    "WMT",
]

# Config Initialization
eval_config = Task2Config(
    model_name=MODEL,
    bnb_config=BitsAndBytesConfig(load_in_8bit=True),  # Enable 8-bit quantization for efficient memory usage
    tickers=STOCK_TICKERS_HIGHEST_CAP_US,
    end_date=END_DATE,
    start_date=START_DATE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(eval_config.model_name)
model = AutoModelForCausalLM.from_pretrained(
    eval_config.model_name,
    quantization_config=eval_config.bnb_config,
    device_map="auto",  # Automatically allocate model across available devices
)

# Evaluation Initialization
# Initialize lists to track trade outcomes and daily returns
trade_win_loss = []  # 1 for win, 0 for loss
daily_returns = []

# Load data
stock_data = pd.read_csv("task2_dsets/test/task2_stocks_test.csv")

# DataFrame for logging cumulative returns based on a threshold trading strategy
# Note: Used for tracking model behavior, not for formal evaluation.
logging_cum_returns_df_threshold_based = []

# Track cumulative returns from the evaluation strategy
eval_cum_returns_data = []

# Evaluation Loop
for date in tqdm(eval_config.eval_dates, desc="Evaluating..."):
    # Filter stock data for the current date
    prices = stock_data[stock_data["Date"] == date._date_repr]

    if prices.empty:
        print(f"No data found for day {date._date_repr}")
        continue

    ticker_signals = {}

    # Generate trading signals for each ticker
    for ticker in eval_config.tickers:
        news = get_news(
            ticker,
            (date - timedelta(days=1))._date_repr,  # Get news from the previous day to prevent post-market close data leakage
            (date - timedelta(days=11))._date_repr,
            "task2_dsets/test/task2_news_test.csv"  # News file
        )

        # Generate evaluation signal using the model
        signal_score = generate_eval_signal(
            tokenizer,
            model,
            device,
            news,
            prices.copy().drop("Future_Close", axis=1)[prices["Ticker"] == ticker],
            eval_config.signal_strength,
            eval_config.threshold,
        )

        ticker_signals[ticker] = signal_score

        # Get today's close price and the future close price for the ticker
        close_price = prices.loc[prices["Ticker"] == ticker, "Close"].item()
        future_price = prices.loc[prices["Ticker"] == ticker, "Future_Close"].item()

        # Threshold-Based Trading (For Model Behavior Analysis)
        # Using generated signal to initiate a trade
        if signal_score >= eval_config.threshold:
            # Long position: Buy at today's close price, sell at future close price
            value_change = (future_price - close_price) / close_price
        elif signal_score <= -1 * eval_config.threshold:
            # Short position: Sell at today's close price, buy back at future close price
            value_change = -1 * ((future_price - close_price) / close_price)
        else:
            # No trade
            value_change = 0

        # Log the results for threshold-based cumulative returns tracking
        logging_cum_returns_df_threshold_based.append({
            "Date": date,
            "Ticker": ticker,
            "ValueChange": value_change,
        })

    # Evaluation Strategy
    # Sort tickers based on signal scores in descending order
    sorted_tickers = sorted(ticker_signals.items(), key=lambda x: x[1], reverse=True)

    # Filter tickers exceeding threshold for long and short positions
    sorted_thresh_tickers_long = [ticker for ticker in sorted_tickers if ticker[1] > eval_config.threshold]
    sorted_thresh_tickers_short = [ticker for ticker in sorted_tickers if ticker[1] < -eval_config.threshold]

    # Select the top N long and short tickers based on configuration
    long_tickers = sorted_thresh_tickers_long[: eval_config.num_long]  # Highest signal scores
    short_tickers = sorted_thresh_tickers_short[: eval_config.num_short]  # Lowest signal scores

    # Initialize a list to store returns for the evaluation day
    eval_rets = []

    # Process selected tickers for trading
    for ticker, signal_score in long_tickers + short_tickers:
        close_price = prices.loc[prices["Ticker"] == ticker, "Close"].item()
        future_price = prices.loc[prices["Ticker"] == ticker, "Future_Close"].item()

        # Calculate value change based on long or short position
        if ticker in dict(long_tickers):
            # Long trade: Buy at close price, sell at future price
            value_change = (future_price - close_price) / close_price
        else:
            # Short trade: Sell at close price, buy back at future price
            value_change = -1 * ((future_price - close_price) / close_price)

        eval_rets.append(value_change)

    # Log cumulative returns and win/loss data for the evaluation day
    eval_cum_returns_data.append({
        "Date": date,
        "MeanEvalReturn": np.mean(eval_rets),  # Average return for the day
        "win_loss": value_change >= 0,  # True if the trade was profitable
    })

# Logging (For Information)
# Create DataFrames
logging_cum_returns_df_threshold_based = pd.DataFrame(logging_cum_returns_df_threshold_based)
logging_cum_returns_df_threshold_based["CumulativeReturn"] = logging_cum_returns_df_threshold_based.groupby("Ticker")[
    "ValueChange"
].cumsum()

# Calculate cumulative returns and win/loss
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
