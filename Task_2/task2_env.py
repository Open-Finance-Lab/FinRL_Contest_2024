from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.config import INDICATORS
from typing import Tuple

import torch
from finrl.meta.preprocessor.preprocessors import data_split

from stable_baselines3.common.env_checker import check_env

from transformers import AutoTokenizer, AutoModelForCausalLM


class Task2Env(gym.Env):
    """A training env for LLM based agents"""

    def __init__(
        self,
        model,
        tokenizer,
        tickers,
        stock_data,
        scale_range: Tuple[int, int],
        max_steps=1, # arbitrary implementation. Feel free to change this.
        threshold=1,
        lookahead=3,  # we set a lookahead of 3 days. This means that the timeframe is long enough for sentiment to effect the market, whilst being short enough that no new sentiment may necessarily overshadow the current trends
    ):

        # self.tokenizer = tokenizer
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model = model

        """observation space defined by natural language input and market data"""
        """observation space = [
                [natural language inputs / news headlines],
                [market data]]"""
        self.observation_space = []
        self.action_space = range(scale_range[0], scale_range[1])
        self.threshold = threshold
        self.max_episode_steps = max_steps
        # process stock_data by ticker to build state
        stock_data["future_close"] = stock_data.groupby("Ticker")["Close"].shift(
            -lookahead
        )
        stock_data = stock_data.dropna(subset=["future_close"])
        self.stock_data = stock_data

        self.date_groups = []

        # Group the data by 'Date' so that we can access all tickers on the same date
        for date, group in stock_data.groupby("Date"):
            self.date_groups.append((date, group))

        """env variables"""
        self.current_step = 0
        self.rewards = []
        self.states = []
        self.cumulative_returns = []

        self.state = self._get_state()

    """State: 
        - List of daily sorted news headlines about stocks
        - Market state"""

    """LLM: 
        - Load model - Llama3 8.1b instruct
        - Use prompting to generate a signal vector using the model
        - """

    """Step: 
        - Parse daily NL headlines -> each day is one string of n headlines
        - pass to AgentLLM to generate signal
        - calculate allocation based off of sentiment vectors
        - calculate reward from prices at set future point"""

    """Notes:
        - the baseline is using a fixed time exit strategy. Actions are generated based off of the top n LLM signals, and are fixed buy or sell
        - You can use the reward from each step for finetuning. """

    def reset(self):
        """reset env"""
        self.current_step = 0
        self.rewards = []
        self.states = []
        self.cumulative_returns = []

        return self._get_state()

    def step(self, actions):
        # actions should be a list of values that go over all tickers
        reward = self._calculate_reward(actions)

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        """update the state at the end of the episode"""
        self._get_state()

        """bookkeeping"""
        self.states.append(self.state)
        self.rewards.append(reward)

        return (
            self.state,
            reward,
            done,
            {},
        )

    def render(self):
        pass

    def _get_state(self):
        """updates and returns self.state"""
        self.state = self.date_groups[self.current_step]
        return self.state

    def _calculate_reward(self, actions):
        """Uses a fixed lookahead to calculate reward based off of a trading action"""
        # use action vector for each stock to determine long, short or pass
        returns = []

        for t in self.state[1].Ticker:
            sentiment_score = actions[t]

            c_price = self.state[1]["Close"]
            f_price = self.state[1]["future_close"]

            if sentiment_score >= self.threshold:
                # long, sell at c price
                value_change = (f_price - c_price) / c_price
            elif sentiment_score <= -1 * self.threshold:
                # short, sell at c price and buy back at f price
                value_change = (c_price - f_price) / f_price
            else:
                value_change = 0

            returns.append(value_change)

        return np.mean(returns)

        # if abs signal is greater than the threshold, then we take up a position and compare the absolute percentage change to future price which is the reward
