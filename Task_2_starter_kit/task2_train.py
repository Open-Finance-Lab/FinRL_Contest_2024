from datetime import datetime, timedelta
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch

from task2_config import Task2Config
from task2_env import Task2Env
from task2_news import get_news
from task2_signal import generate_signal


# Constants
END_DATE = "2023-12-16"
START_DATE = "2020-01-01"

STOCK_TICKERS_HIGHEST_CAP_US = [
    "AAPL",
    "NVDA",
    "GOOG",
    "AMZN",
    "MSFT",
    "XOM",
    "WMT",
]


def setup_model_config():
    """Setup model configurations for training.
    
    Returns:
        Tuple of (bnb_config_4bit, bnb_config_8bit, device).
    """
    bnb_config_4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4",
    )

    bnb_config_8 = BitsAndBytesConfig(load_in_8bit=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return bnb_config_4, bnb_config_8, device


def initialize_model(config, device):
    """Initialize and configure the model and tokenizer.
    
    Args:
        config: Training configuration object.
        device: torch device to use.
    
    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config_4,
        device_map="auto",
    )

    # Configure model settings
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Setup LoRA configuration
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
    
    return tokenizer, model


def plot_training_metrics(losses, rewards, returns, running_eval):
    """Plot and save training metrics.
    
    Args:
        losses: List of training losses.
        rewards: List of rewards.
        returns: List of returns.
        running_eval: List of running evaluation values.
    """
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


def main():
    """Main training loop."""
    # Initialize training configuration
    train_config = Task2Config(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        bnb_config=BitsAndBytesConfig(load_in_8bit=True),
        tickers=STOCK_TICKERS_HIGHEST_CAP_US,
        end_date=END_DATE,
        start_date=START_DATE,
        lookahead=3,
        signal_strength=10,
        max_train_steps=50,
    )

    # Load data
    stock_data = pd.read_csv("task2_stocks.csv")

    # Setup model and environment
    bnb_config_4, bnb_config_8, device = setup_model_config()
    tokenizer, model = initialize_model(train_config, device)

    task2env = Task2Env(
        model,
        tokenizer,
        STOCK_TICKERS_HIGHEST_CAP_US,
        stock_data,
        (-2, 2),
        max_steps=252 - 4,
        lookahead=14,
    )

    # Initialize training metrics
    state = task2env.reset()
    actions = []
    rewards = []
    returns = []
    running_eval = []
    losses = []

    optimizer = Adam(model.parameters(), lr=1e-5)

    # Training loop
    for step in tqdm(
        range(train_config.max_train_steps),
        desc=f"training for max train steps: {train_config.max_train_steps}"
    ):
        date, prices = state
        date = pd.Timestamp(date)
        ticker_actions = {}
        log_probs = []
        rewards_list = []
        
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
                train_config.signal_strength,
                train_config.threshold,
            )
            ticker_actions[t] = sentiment_score
            log_probs.append(log_prob)

        state, reward, done, d = task2env.step(ticker_actions)
        
        # Update metrics
        actions.append(ticker_actions)
        rewards.append(reward)
        returns.append(d["price change"])
        running_eval.append(d["running eval"])
        rewards_list.append(reward)

        # Compute and apply gradients
        loss = -torch.stack(log_probs) * torch.tensor(reward)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if done:
            break

    # Plot results
    plot_training_metrics(losses, rewards, returns, running_eval)


### WHEN TO RUN MAIN