# FinRL Task 2

This task aims to develop LLMs that can generate and engineer effective signals from news by using Reinforcement Learning from Market Feedback (RLMF).

In this task, the LLM will be used to generate one type of signal (e.g., a sentiment score) based on the content of news. Contestants will develop models that leverage RLMF to adjust the signals based on the market feedback. 

The LLM should be loadable with huggingface and do inference on 20GB GPU.

The quality of the trading signals will be evaluated with a fixed time exit strategy that can either hold, long or short a given stock.  

## Datasets

`task2_dsets.zip` contains:
- A 70-30 data split into a test and train directory
- Each split has: 
    - `task2_stocks_{split}.csv`: an OHLCV dataset of a list of stocks.
    - `task2_news_{split}.csv`: a corresponding news dataset for the list of stocks.

Contestants are free to use external datasets to deveop RLMF methods and fine-tune the LLMs. The evaluation phase will use the testing datasets with the same fields as the dataset provided. 

## Starter Kit Descriptions
We use the sentiment score as an example in our starter kit. You can improve on this sentiment analysis or generate your own signal. 

The starter kit includes:
- `task2_dsets.zip`
- `task2_env.py`: an example of environment to utilize RLMF. It defines the state and action space for the LLM agent to explore. This environment generates reward signals based on the stock price movement (market feedback) and sentiment score (LLM output). You are free to make changes or write your own environment.
- `task2_news.py`: it contains the function to read the news dataset.
- `task2_signal.py`: it contains the prompt and code to use the LLM to generate sentiment score. In this file you may change the prompt to generate various other signals. For example you can change the range of sentiment scores as long as the thresholds are set to 30% of the maximum and minimum value. (We will evaluate using these thresholds so please do not use other values)
- `task2_train.py`: it imports a model and tokenizer. It calls helper functions from the other task2 files to fetch stock tickers, fetch appropriate news to tickers, use the environment in order to fine-tune the LLM.
- `task2_eval.py`: we provide a sample evaluation file that shows how you can evaluate your model using a simple strategy.
- `task2_config.py`: We provide a configuration class that is used for both training and evaluation. Both the train and evaluation file use this class. Not all parameters need to be used in training or evaluation, and we describe their uses in detail in the enxt section.

parameters you may set freely:
- `END_DATE`: you may set your own training ranges. 
- `START_DATE`
- `signal_strength` You can set a custom signal range for your model to learn to estimate.
- datasets: you may train the model on additional datasets using RLMF, eval set will use a larger hidden testing set similar to the split given
- signals: we provide a sample signal generation function. As with the training, you are free to provide your own signal generation function
- `train steps`: you may change the number of training steps that your model does. We set this to 50 in the demo kit, but we recommend setting it to at least the length of your training data.

## Setup Instructions

To get started, follow these steps:

1. **Create a Virtual Environment** (optional but recommended):
   It's highly recommended to set up a virtual environment to avoid conflicts with other packages. You can do this using `virtualenv`:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install Dependencies**:
   Once the virtual environment is set up, install the required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Access to Model**:
   Make sure you have access to the desired model on Hugging Face (e.g., Llama or your preferred model). You may need to log in to your Hugging Face account and accept terms if necessary. 

   Then, you should be able to load the model as specified in the code.

## Evaluation Template
You may use this template to evaluate your models in a framework aligned to how we will be evaluating your submissions. We provide some hyperparameters and configuration settings to demonstrate how to use the template. The template uses a config class to localize configurations for your convenience. You may create a validation set from the training data and use that with the evaluation class to test your model on out of sample data. 

Your model will generate a signal for each ticker each timestep which will be used in the evaluation strategy to open simualted positions. The demo code tracks returns, cumulative returns and win/loss rate, which are important metrics that allow us to compute many other metrics from them. You may use additional metrics in your own evaluation to better understand how your models are performing. 

The evaluation kit includes:
- `task2_eval.py`: this file contains a configuration class, code to generate signals and evaluate the model performance and code to log model performance
- model loader: we show how to use huggingface to load models 

### Evaluation strategy
The evaluation strategy opens a long and short position on the 3 stocks with the strongest positive and negative signals respectively in the following pattern: 
- Your model will predict a signal as a numerical value in the range that you provide
- Our strategy takes the top 3 and bottom 3 signal scores and checks if they exceed the thresholds. (Positive signal should exceed the 30% boundary, and negative signal should be less than the -30% boundary)
- The strategy will then long and short these stocks depending on their signal score with a fixed lookahead of 3 days.

## Evaluation and Submission guidelines
Please submit your model and appropriate tokenizer, as well as any files that are needed to run your submission. Evaluation will be done by importing your submissions into the evaluation file to execute all submissions on an identical platform. We ask contestants to therefore submit all of their relevant materials whilst maintaining the endpoints in the other task2 files to facilitate testing.

Please provide a readme that describes your submission and explains important things to note when running it so that we can ensure we run your submission as it was intended.

```
├── finrl-contest-task-2 
│ ├── trained_models # Your fine-tuned LLM weights.
│ ├── task2_signal.py # File for your signal 
│ ├── task2_eval.py # File to load your model, tokenizer, and prompt for evaluation.
│ ├── task2_config.py # File for the evaluation configuration.
│ ├── task2_env.py # File for environment, this should include your reward function.
│ ├── readme.md # File to explain the your code
│ ├── requirements.txt # Have it if adding any new packages
│ ├── And any additional scripts you create
```