# Data-centric Stock Trading Task
This template includes all the essential source files for the Stock Trading task in the [FinRL Contest 2023](https://open-finance-lab.github.io/finrl-contest.github.io/). Below, you'll find detailed descriptions of each component.

## Instruction
To install the necessary dependencies, please ensure you are using Python 3.10 as your interpreter.
```
pip install swig
pip install box2d
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```

## Deliverable
We encourage contestants to curate their own dataset. To ensure a smooth testing phase, participants should design their data curation pipeline to seamlessly incorporate previously unseen, hidden test data (provided in a basic format as `train_data.csv`). For instance, we will use `test.py` to assess the performance of your submitted model.

```
python3 test.py --start_date 2022-01-01 --end_date 2022-12-31 --data_file test_data.csv
```

## Data

We offer a selection of fundamental trading indicators. Their use is optional based on your preference. Contestants are free to design data processing strategies and perform feature engineering, such as constructing new indicators based on existing and/or external market data.

1. **MACD (Moving Average Convergence Divergence)**
   - A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.

2. **Bollinger Bands Upper Band (boll_ub)**
   - Represents the upper threshold of the Bollinger Bands, which is typically two standard deviations above the 20-day simple moving average (SMA).

3. **Bollinger Bands Lower Band (boll_lb)**
   - Represents the lower threshold of the Bollinger Bands, which is typically two standard deviations below the 20-day SMA.

4. **RSI 30 (Relative Strength Index for 30 periods)**
   - A momentum oscillator that measures the speed and change of price movements. RSI oscillates between zero and 100.

5. **CCI 30 (Commodity Channel Index for 30 periods)**
   - A versatile indicator that can be used to identify a new trend or warn of extreme conditions. It measures the current price level relative to an average price level over a given period of time.

6. **DX 30 (Directional Movement Index for 30 periods)**
   - An indicator that identifies whether a security is trending. It does this by comparing highs and lows over time.

7. **Close 30 SMA (30-period Simple Moving Average of Closing Prices)**
   - Represents the average closing price over the last 30 periods.

8. **Close 60 SMA (60-period Simple Moving Average of Closing Prices)**
   - Represents the average closing price over the last 60 periods.

9. **VIX (Volatility Index)**
   - Often referred to as the "fear index", it represents the market's expectation of 30-day forward-looking volatility. It is calculated from the prices of selected stock option contracts on the S&P 500 Index.

10. **Turbulence**
   - To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
