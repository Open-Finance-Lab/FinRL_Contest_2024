from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Tuple


class Task2Env(gym.Env):
    """A training env for LLM based agents.
    
    State includes:
        - List of daily sorted news headlines about stocks
        - Market state
    
    LLM:
        - Load model - Llama3 8.1b instruct
        - Use prompting to generate a signal vector using the model
    
    Step:
        - Parse daily NL headlines -> each day is one string of n headlines
        - pass to AgentLLM to generate signal
        - calculate allocation based off of sentiment vectors
        - calculate reward from prices at set future point
    
    Notes:
        - The baseline is using a fixed time exit strategy. Actions are generated 
          based off of the top n LLM signals, and are fixed buy or sell
        - You can use the reward from each step for finetuning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        tickers,
        stock_data,
        scale_range: Tuple[int, int],
        max_steps: int = 1,  # arbitrary implementation. Feel free to change this
        threshold: int = 3,
        # we set a lookahead of 3 days. This means that the timeframe is long enough
        # for sentiment to effect the market, whilst being short enough that no new
        # sentiment may necessarily overshadow the current trends
        lookahead: int = 3,
    ):
        """Initialize Task2Env.
        
        Args:
            model: The LLM model to use.
            tokenizer: Tokenizer for the model.
            tickers: List of stock tickers.
            stock_data: DataFrame containing stock price data.
            scale_range: Tuple defining min/max signal range.
            max_steps: Maximum steps per episode. Defaults to 1.
            threshold: Signal threshold for taking action. Defaults to 3.
            lookahead: Days to look ahead for returns. Defaults to 3.
        """
        self.observation_space = []
        self.action_space = range(scale_range[0], scale_range[1])
        self.threshold = threshold
        self.max_episode_steps = max_steps

        # Process stock_data by ticker to build state
        stock_data["future_close"] = stock_data.groupby("Ticker")["Close"].shift(
            -lookahead)
        stock_data = stock_data.dropna(subset=["future_close"])
        self.stock_data = stock_data

        self.date_groups = []

        # Group the data by 'Date' so that we can access all tickers on the same date
        for date, group in stock_data.groupby("Date"):
            self.date_groups.append((date, group))

        # Environment variables
        self.current_step = 0
        self.rewards = []
        self.states = []
        self.cumulative_returns = []

        self.state = self._get_state()

        # Reward hyperparams - you may set these differently through the env constructor
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

        # Evaluation amount
        self.eval_amt = 1e6

    def reset(self):
        """Reset environment state.
        
        Returns:
            Initial state of the environment.
        """
        self.current_step = 0
        self.rewards = []
        self.states = []
        self.cumulative_returns = []

        return self._get_state()

    def step(self, actions):
        """Take environment step based on actions.
        
        Args:
            actions: List of sentiment values for each ticker.

        Returns:
            Tuple containing:
                - Next state
                - Reward
                - Done flag
                - Info dictionary with price changes and evaluation metrics
        """
        sum_reward, p_return = self._calculate_reward(actions)
        running_eval = self._evaluate_model(actions)

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        # Update the state at the end of the episode
        self._get_state()

        # Bookkeeping
        self.states.append(self.state)
        self.rewards.append(sum_reward)

        return (
            self.state,
            sum_reward,
            done,
            {"price change": p_return, "running eval": running_eval},
        )

    def render(self):
        """Render environment state."""
        pass

    def _get_state(self):
        """Update and return environment state.
        
        Returns:
            Current state of the environment.
        """
        self.state = self.date_groups[self.current_step]
        return self.state

    def _calculate_reward(self, actions):
        """Calculate reward based on actions and price movements.
        
        Uses a fixed lookahead to calculate reward based on the stock price movement.
        Demonstrates a simple mechanism that takes model confidence into account.
        You can design your own reward function.

        Args:
            actions: Dictionary mapping tickers to sentiment scores.

        Returns:
            Tuple of (total_reward, mean_price_return).
        """
        # Use action vector for each stock to determine long, short or pass
        prices = self.state[1]
        p_returns = []
        rewards = []

        for ticker in prices.Ticker:
            sentiment_score = actions[ticker]
            c_price = prices.loc[prices["Ticker"] == ticker, "Close"].values[0]
            f_price = prices.loc[prices["Ticker"] == ticker, "future_close"].values[0]

            value_change = (f_price - c_price) / c_price

            if sentiment_score >= self.threshold:
                # Long position
                if value_change > self.strong_positive_return:
                    reward = (
                        self.great_positive_reward 
                        if sentiment_score > self.high_confidence 
                        else self.positive_reward
                    )
                elif value_change < 0:  # Negative return despite positive sentiment
                    reward = (
                        self.highest_negative_reward
                        if sentiment_score > self.high_confidence
                        else self.negative_reward
                    )
                else:
                    reward = (
                        self.weak_positive_reward 
                        if sentiment_score > self.high_confidence 
                        else self.moderate_reward
                    )

            elif sentiment_score <= -1 * self.threshold:
                # Short position
                if value_change < -self.strong_negative_return:
                    reward = (
                        self.highest_negative_reward
                        if sentiment_score > self.high_confidence
                        else self.negative_reward
                    )
                elif value_change > 0:  # Positive return despite negative sentiment
                    reward = (
                        self.great_positive_reward 
                        if sentiment_score > self.high_confidence 
                        else self.positive_reward
                    )
                else:
                    reward = (
                        self.weak_negative_reward
                        if sentiment_score > self.high_confidence
                        else self.moderate_negative_reward
                    )
            else:
                reward = self.passive_reward  # No strong action taken

            rewards.append(reward)
            p_returns.append(value_change)

        return (np.sum(rewards), np.mean(p_returns))

    def _evaluate_model(self, actions):
        """Evaluate model performance using a simple trading strategy.
        
        A simple strategy to evaluate the LLM-generated sentiment score.
        Uses a fixed lookahead to calculate return based off of a trading action.
        You can design your own trading strategy here to evaluate your signal.
        In the contest evaluation phase, we will use a similar trading strategy.

        Args:
            actions: Dictionary mapping tickers to sentiment scores.

        Returns:
            Current evaluation amount after applying returns.
        """
        returns = []
        prices = self.state[1]
        
        for ticker in self.state[1].Ticker:
            sentiment_score = actions[ticker]

            c_price = prices.loc[prices["Ticker"] == ticker, "Close"].values[0]
            f_price = prices.loc[prices["Ticker"] == ticker, "future_close"].values[0]

            if sentiment_score >= self.threshold:
                # Long position
                value_change = (f_price - c_price) / c_price
            elif sentiment_score <= -1 * self.threshold:
                # Short position
                value_change = (f_price - c_price) / c_price
            else:
                value_change = 0

            returns.append(value_change)

        avg_return = np.mean(returns)
        self.eval_amt = self.eval_amt * (1 + avg_return)
        return self.eval_amt