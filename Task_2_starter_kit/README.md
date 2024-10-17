# FinRL task 2
This task aims to develop LLMs that can generate and engineer effective signals from news by using Reinforcement Learning from Market Feedback (RLMF).

In this task, the LLM will be used to generate one type of signal (e.g., a sentiment score) based on the content of news. Contestants will develop models that leverage RLMF to adjust the signals based on the market feedback. 

The LLM should be loadable with huggingface and do inference on 20GB GPU.

The quality of the trading signals will be evaluated with a fixed time exit strategy that can either hold, long or short a given stock.  


## Datasets

`task2_dataset.zip` contains:
- `task2_stocks.csv`: an OHLCV dataset of a list of stocks.
- `task2_news.csv`: a corresponding news dataset for the list of stocks.

Contestants are free to use external datasets to deveop RLMF methods and fine-tune the LLMs. The evaluation phase will use the testing datasets with the same fields as the dataset provided. 

## Starter kit description
We use the sentiment score as an example in our starter kit. You can improve on this sentiment analysis or generate your own signal. 

The starter kit includes:
- `task2_dataset.zip`
- `task2_env.py`: an example of environment to utilize RLMF. It defines the state and action space for the LLM agent to explore. This environment generates reward signals based on the stock price movement (market feedback) and sentiment score (LLM output). You are free to make changes or write your own environment.
- `task2_news.py`: it contains the function to read the news dataset.
- `task2_stocks.py`: it contains the function to dowload the OHLCV dataset from yfinance.
- `task2_signal.py`: it contains the prompt and code to use the LLM to generate sentiment score.
- `task2_train.py`: it imports a model and tokenizer. It calls helper functions from the other task2_ files to fetch stock tickers, fetch appropriate news to tickers, use the environment in order to fine-tune the LLM.

We will provide the evaluation code soon.

## Evaluation and Submission guidelines
Please submit your model and appropriate tokenizer, as well as any files that are needed to run your submission. Evaluation will be done by importing your submissions into the evaluation file to execute all submissions on an identical platform. We ask contestants to therefore submit all of their relevant materials whilst maintaining the endpoints in the other task2 files to facilitate testing.

Please provide a readme that describes your submission and explains important things to note when running it so that we can ensure we run your submission as it was intended.

```
├── finrl-contest-task-2 
│ ├── trained_models # Your fine-tuned LLM weights.
│ ├── task2_signal.py # File for your signal 
│ ├── task2_test.py # File to load your model, tokenizer, and prompt for evaluation. We will provide an example file soon.
│ ├── task2_env.py # File for environment, this should include your reward function. 
│ ├── readme.md # File to explain the your code
│ ├── requirements.txt # Have it if adding any new packages
│ ├── And any additional scripts you create
```


