# FinRL Contest 2024
This repository contains the starter kit and tutorials for the ACM ICAIF 2024 FinRL Contest.


## Outline
  - [Tutorial](#tutorial)
  - [Task 1 Starter Kit](#task-1-starter-kit)
  - [Task 2 Starter Kit](#task-2-starter-kit)
  - [Report Submission Requirement](#report-submission-requirement)
  - [Resources](#resources)


## Tutorial
| Task | Model | Environment | Dataset | Link |
| ---- |------ | ----------- | ------- | ---- |
| Stock Trading | PPO | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/blob/main/Tutorials/FinRL_stock_trading_demo.ipynb) |
| Stock Trading @ [FinRL Contest 2023](https://open-finance-lab.github.io/finrl-contest.github.io/)| PPO | Stock Trading Environment | OHLCV | [Baseline solution](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Tutorials/FinRL_Contest_2023_Task_1_baseline_solution) |
| Stock Trading | Ensemble | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/blob/main/Tutorials/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) for [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)|
| Task 1 Crypto Trading Ensemble (requires the files in the starter kit to run) | Ensemble | Crypto Trading Environment | BTC LOB 1sec | [Code](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Tutorials/Task_1_tutorial) |
| Sentiment Analysis with Market Feedback | ChatGLM2-6B | -- | Eastmoney News | [Code](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Sentiment_Analysis_v1/FinGPT_v1.0) |




## Task 1 Starter Kit

Please see [Task_1_starter_kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit) folder.

**New notes for clarifications**

The basic requirement is that your model should be able to interact with the environment. The code for training agent and ensemble is just an example solution for your reference.

1. You are free to apply any method for ensemble learning. (For example, You can add new agents, use different ensemble algorithms, adjust hyperparameters, etc) The code provided is just to help get started and we encourage innovation.
2. You are not required to stick to the 8 features we provide. But for evaluation purpose, please make sure that your new technical factors, if any, can be calculated based on the unseen data. Please include this code and state clearly in readme.
3. We will use the provided environment to evaluate. So it is not encouraged to change the existing parameters in the environment. However, you can fully utilize the environment settings and the massively parallel simulation.
4. To encourage innovation, if you want to add new mechanisms or use the unused settings (e.g. short sell) in the environment, please also submit your environment, ensure it works with your agent for evaluation, and describe the new changes in the readme.



## Task 2 Starter Kit

Please see [Task_2_starter_kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_2_starter_kit) folder.


## Report Submission Requirement
Each team should also submit a 1-2 page report for the corresponding task they choose with the [ACM sigconf template](https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc) through Open Review. The title should start with “FinRL Contest 2024 Task I” or “FinRL Contest 2024 Task II.”

## Resources
Useful materials and resources for contestants to learn about FinRL, including APIs and more tutorials:
* [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
* [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)
* [FinRL Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials)



