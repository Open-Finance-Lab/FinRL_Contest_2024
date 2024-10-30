import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from peft import get_peft_model, LoraConfig, TaskType

from task2_stocks import get_stock_data
from task2_news import get_news
from task2_signal import generate_signal
from task2_config import Task2Config

from task2_config import Task2Config


# Date ranges for the starter solution
END_DATE = "2023-12-16"
START_DATE = "2020-01-01"

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

train_config = Task2Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    bnb_config=BitsAndBytesConfig(load_in_8bit=True),
    tickers=STOCK_TICKERS_HIGHEST_CAP_US,
    end_date=END_DATE,
    start_date=START_DATE,
    lookahead=3,
    signal_strengh=10,
    max_train_steps=50,
)


"""load data and make dset - first we load in the ticker data for each ticker, then we enrich that with news data"""
# stock_data = get_stock_data(STOCK_TICKERS_HIGHEST_CAP_US, START_DATE, END_DATE)
stock_data = pd.read_csv("task2_stocks.csv")

"""load model and env"""
from task2_env import Task2Env

bnb_config_4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.bfloat16
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="fp4",  # 'nf4' or 'fp4'
)

bnb_config_8 = BitsAndBytesConfig(load_in_8bit=True)


num_gpus = torch.cuda.device_count()
max_memory = {i: "22GiB" for i in range(num_gpus)}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # map to auto for multi gpu

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
model = AutoModelForCausalLM.from_pretrained(
    train_config.model_name,
    quantization_config=bnb_config_4,
    # quantization_config=bnb_config_4,
    device_map="auto",
    # max_memory=max_memory,
)
# model = torch.nn.DataParallel(model)
# model.to(device)

# grad checkpntg
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# LoRA config to run on smaller gpus
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=30,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

task2env = Task2Env(
    model,
    tokenizer,
    STOCK_TICKERS_HIGHEST_CAP_US,
    stock_data,
    (-2, 2),
    max_steps=252 - 4,
    lookahead=14,
)


state = task2env.reset()
actions = []
rewards = []
returns = []
running_eval = []
losses = []

optimizer = Adam(model.parameters(), lr=1e-5)

# you can also set this to true or while not horizon len met
for step in tqdm(
    range(train_config.max_train_steps), desc=f"training for max train steps: {train_config.max_train_steps}"
):
    date, prices = state
    date = pd.Timestamp(date)
    ticker_actions = {}
    log_probs = []
    rewards_list = []
    done = False

    for t in prices.Ticker:
        news = get_news(
            t,
            (date - timedelta(days=1))._date_repr,
            (date - timedelta(days=11))._date_repr,
            "task2_news.csv",
        )
        sentiment_score, log_prob = generate_signal(
            tokenizer,
            model,
            device,
            news,
            prices.copy().drop("future_close", axis=1)[prices["Ticker"] == t],
            train_config.signal_strengh,
            train_config.threshold,
        )
        ticker_actions[t] = sentiment_score
        log_probs.append(log_prob)

    """actions may look like: {'AAPL': 1.0, 'NVDA': 1.5, 'GOOG': 0, 'AMZN': 1.5, 'MSFT': -1.5, 'XOM': 0.5, 'WMT': 1.5}"""
    state, reward, done, d = task2env.step(ticker_actions)
    actions.append(ticker_actions)
    rewards.append(reward)
    returns.append(d["price change"])
    running_eval.append(d["running eval"])
    rewards_list.append(reward)

    loss = -torch.stack(log_probs) * torch.tensor(reward)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if done:
        break


# Create subplots
plt.figure(figsize=(14, 10))

# Plot Loss over Time
plt.subplot(2, 2, 1)
plt.plot(losses, label="Loss", color="red")
plt.title("Loss over Time")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()

# Plot Reward over Time
plt.subplot(2, 2, 2)
plt.plot(rewards, label="Reward", color="blue")
plt.title("Reward over Time")
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.legend()

# Plot Average Price Change over Time
plt.subplot(2, 2, 3)
plt.plot(returns, label="Average Price Change", color="green")
plt.title("Average Price Change over Time")
plt.xlabel("Training Step")
plt.ylabel("Average Price Change")
plt.legend()

# Plot Running Evaluation Amount over Time
plt.subplot(2, 2, 4)
plt.plot(running_eval, label="Running Evaluation Amount", color="purple")
plt.title("Running Evaluation Amount over Time")
plt.xlabel("Training Step")
plt.ylabel("Evaluation Amount")
plt.legend()

plt.tight_layout()
plt.savefig("training.png")
