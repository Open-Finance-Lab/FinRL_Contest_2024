import pandas as pd


class Task2Config:
    """configuration class"""

    def __init__(
        self,
        model_name,
        bnb_config,
        tickers,
        end_date="",
        start_date="",
        lookahead=3,
        signal_strengh=10,
        num_short=3,
        num_long=3,
        max_train_steps=50
    ):
        self.model_name = model_name
        self.bnb_config = bnb_config
        self.tickers = tickers
        self.end_date = end_date
        self.start_date = start_date
        self.eval_dates = pd.bdate_range(start=start_date, end=end_date)  # Only market open days
        self.lookahead = lookahead
        self.signal_strengh = signal_strengh
        self.threshold = signal_strengh // 3  # 30% as a threshold
        self.num_short = num_short
        self.num_long = num_long
        self.max_train_steps = max_train_steps
