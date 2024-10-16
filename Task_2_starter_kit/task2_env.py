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
        max_steps=1,  # arbitrary implementation. Feel free to change this.
        threshold=3,
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
        stock_data["future_close"] = stock_data.groupby("Ticker")["Close"].shift(-lookahead)
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

        # reward hyperparams - you may set these differently through the env constructor
        self.strong_positive_return = 0.02  # Above 2% return considered strong positive
        self.strong_negative_return = -0.02  # Below -2% return considered strong negative

        self.great_positive_reward = 2.0  # High reward for correct confident decisions
        self.positive_reward = 1.0  # Standard reward for correct confident decisions
        self.weak_positive_reward = 0.5  # Lower reward for weak but correct decisions

        self.highest_negative_reward = -2.0  # Strong penalty for confident wrong decisions
        self.negative_reward = -1.0  # Standard penalty for wrong decisions
        self.weak_negative_reward = -0.5  # Lesser penalty for less confident wrong decisions

        self.high_confidence = 6  # Confidence level above which actions are considered highly confident
        self.passive_reward = -0.001  # Neutral reward for no action taken
        self.moderate_reward = 0.1  # Moderate reward for close-to-neutral actions
        self.moderate_negative_reward = -0.1  # Small negative reward for minor wrong actions

        # eval
        self.eval_amt = 1e6

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

        sum_reward, p_return = self._calculate_reward(actions)
        running_eval = self._evaluate_model(actions)

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        """update the state at the end of the episode"""
        self._get_state()

        """bookkeeping"""
        self.states.append(self.state)
        self.rewards.append(sum_reward)

        return (
            self.state,
            sum_reward,
            done,
            {"price change": p_return, "running eval": running_eval},
        )

    def render(self):
        pass

    def _get_state(self):
        """updates and returns self.state"""
        self.state = self.date_groups[self.current_step]
        return self.state

    def _calculate_reward(self, actions):
        """
        Uses a fixed lookahead to calculate reward based on the stock price movement. 
        We demonstrate a simple mechanism that takes model confidence into account.
        You can design your own reward function.
        """
        # use action vector for each stock to determine long, short or pass
        prices = self.state[1]
        p_returns = []
        rewards = []

        for ticker in prices.Ticker:
            sentiment_score = actions[ticker]
            c_price = prices.loc[prices["Ticker"] == ticker, "Close"].values[0]
            f_price = prices.loc[prices["Ticker"] == ticker, "future_close"].values[0]

            value_change = (f_price - c_price) / c_price

            if sentiment_score >= self.threshold:
                # long, sell at c price
                if value_change > self.strong_positive_return:
                    reward = (
                        self.great_positive_reward if sentiment_score > self.high_confidence else self.positive_reward
                    )
                elif value_change < 0:  # Negative return despite positive sentiment
                    reward = (
                        self.highest_negative_reward
                        if sentiment_score > self.high_confidence
                        else self.negative_reward
                    )
                else:
                    reward = (
                        self.weak_positive_reward if sentiment_score > self.high_confidence else self.moderate_reward
                    )

            elif sentiment_score <= -1 * self.threshold:
                # short, sell at c price and buy back at f price
                if value_change < -self.strong_negative_return:
                    reward = (
                        self.highest_negative_reward
                        if sentiment_score > self.high_confidence
                        else self.negative_reward
                    )
                elif value_change > 0:  # Positive return despite negative sentiment
                    reward = (
                        self.great_positive_reward if sentiment_score > self.high_confidence else self.positive_reward
                    )
                else:
                    reward = (
                        self.weak_negative_reward
                        if sentiment_score > self.high_confidence
                        else self.moderate_negative_reward
                    )
            else:
                reward = self.passive_reward  # no strong action taken

            rewards.append(reward)
            p_returns.append(value_change)

        return (np.sum(rewards), np.mean(p_returns))

    def _evaluate_model(self, actions):
        """
        A simple strategy to evaluate the LLM-generated sentiment score. 
        Uses a fixed lookahead to calculate return based off of a trading action.
        Feel free to design your own trading strategy here to evaluate your signal.
        In the contest evaluation phase, we will use a similar trading strategy to evalute signals.
        """
        # use action vector for each stock to determine long, short or pass
        returns = []
        prices = self.state[1]
        for ticker in self.state[1].Ticker:
            sentiment_score = actions[ticker]

            c_price = prices.loc[prices["Ticker"] == ticker, "Close"].values[0]
            f_price = prices.loc[prices["Ticker"] == ticker, "future_close"].values[0]

            if sentiment_score >= self.threshold:
                # long, sell at c price
                value_change = (f_price - c_price) / c_price
            elif sentiment_score <= -1 * self.threshold:
                # short, sell at c price and buy back at f price
                value_change = (f_price - c_price) / c_price

            else:
                value_change = 0

            returns.append(value_change)

        avg_return = np.mean(returns)
        self.eval_amt = self.eval_amt * (1 + avg_return)
        return self.eval_amt
        # if abs signal is greater than the threshold, then we take up a position and compare the absolute percentage change to future price which is the reward
