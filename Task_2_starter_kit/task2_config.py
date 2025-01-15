import pandas as pd


class Task2Config:
    """Configuration class for Task 2.

    This class encapsulates configuration parameters, including model details,
    market tickers, date ranges, and training settings.

    Attributes:
        model_name (str): The name of the model to be used.
        bnb_config (dict): Configuration for Bayesian optimization (or other relevant purpose).
        tickers (list of str): List of stock tickers to analyze.
        end_date (str): The end date for the data range (YYYY-MM-DD).
        start_date (str): The start date for the data range (YYYY-MM-DD).
        eval_dates (pd.DatetimeIndex): Business days within the date range.
        lookahead (int): Number of days to look ahead for predictions.
        signal_strength (int): The strength of the signal for generating predictions.
        threshold (int): Threshold for signal filtering (30% of signal strength).
        num_short (int): Number of short positions to consider.
        num_long (int): Number of long positions to consider.
        max_train_steps (int): Maximum number of training steps for the model.
    """

    def __init__(
        self,
        model_name: str,
        bnb_config: dict,
        tickers: list,
        end_date: str = "",
        start_date: str = "",
        lookahead: int = 3,
        signal_strength: int = 10,
        num_short: int = 3,
        num_long: int = 3,
        max_train_steps: int = 50
    ):
        """Initialize the Task2Config class.

        Args:
            model_name: The name of the model to be used.
            bnb_config: Configuration for Bayesian optimization (or other relevant purpose).
            tickers: List of stock tickers to analyze.
            end_date: The end date for the data range (YYYY-MM-DD).
            start_date: The start date for the data range (YYYY-MM-DD).
            lookahead: Number of days to look ahead for predictions. Defaults to 3.
            signal_strength: The strength of the signal for generating predictions.
                Defaults to 10.
            num_short: Number of short positions to consider. Defaults to 3.
            num_long: Number of long positions to consider. Defaults to 3.
            max_train_steps: Maximum number of training steps for the model.
                Defaults to 50.
        """
        self.model_name = model_name
        self.bnb_config = bnb_config
        self.tickers = tickers
        self.end_date = end_date
        self.start_date = start_date
        # Only market open days
        self.eval_dates = pd.bdate_range(start=start_date, end=end_date)
        self.lookahead = lookahead
        self.signal_strength = signal_strength
        # 30% as a threshold
        self.threshold = signal_strength // 3
        self.num_short = num_short
        self.num_long = num_long
        self.max_train_steps = max_train_steps