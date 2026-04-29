# GreekFire Experiment Log
This file tracks the results of walk-forward backtesting experiments.

## Run 1 — 2026-04-27
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: Random Forest
- n_subfeatures: 4
- max_depth: 10
- CAGR: 2.42%
- Sharpe Ratio: -0.18
- Max Drawdown: 34.07%
- Win Rate: 55.67%
- Avg Excess Return: 0.24%
- Top 3 features: macd_signal (0.0884), iv (0.0879), vega (0.0785)
- Notes: 

## Run 2 — 2026-04-28
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: Random Forest
- n_subfeatures: 4
- max_depth: 10
- CAGR: -5.55%
- Sharpe Ratio: -0.79
- Max Drawdown: 56.64%
- Win Rate: 50.0%
- Avg Excess Return: -0.31%
- Top 3 features: atr_norm (0.094), macd_signal (0.09), iv_rank (0.0786)
- Notes: 

## Run 3 — 2026-04-28
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: XgBoost
- n_subfeatures: 4
- max_depth: 10
- CAGR: 3.29%
- Sharpe Ratio: -0.07
- Max Drawdown: 46.05%
- Win Rate: 60.31%
- Avg Excess Return: 0.35%
- Top 3 features: macd_signal (0.0895), iv (0.0885), atr_norm (0.0871)
- Notes: 

## Run 4 — 2026-04-28
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: Random Forest plus XGboost averaged out
- n_subfeatures: 4
- max_depth: 10
- CAGR: 3.64%
- Sharpe Ratio: -0.01
- Max Drawdown: 37.82%
- Win Rate: 57.73%
- Avg Excess Return: 0.4%
- Top 3 features: atr_norm (0.0912), macd_signal (0.0896), iv (0.084)
- Notes: 

## Run 5 — 2026-04-29
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: Ensemble (RF + XGBoost, 50/50 weighted average)
- Training Window: 252 days rolling
- Test Step: 5 days
- Top N Stocks: 5
- RF n_trees: 100
- RF n_subfeatures: 4
- RF max_depth: 10
- XGBoost num_round: 300
- XGBoost max_depth: 3
- XGBoost eta (learning rate): 0.03
- XGBoost min_child_weight: 3
- XGBoost gamma: 0.1
- XGBoost reg_lambda: 5
- XGBoost subsample: 0.7
- XGBoost colsample_bytree: 0.7
- CAGR: 2.11%
- Sharpe Ratio: 0.24
- Max Drawdown: 29.72%
- Win Rate: 59.66%
- Avg Excess Return: 0.58%
- Top 3 features: iv_rank (0.1425), atr_norm (0.0758), vol_ratio (0.0707)
- Notes: 

## Run 6 — 2026-04-29
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: Ensemble (RF + XGBoost, 50/50 weighted average)
- Training Window: 252 days rolling
- Test Step: 5 days
- Top N Stocks: 5
- RF n_trees: 100
- RF n_subfeatures: 4
- RF max_depth: 10
- XGBoost num_round: 300
- XGBoost max_depth: 3
- XGBoost eta (learning rate): 0.03
- XGBoost min_child_weight: 3
- XGBoost gamma: 0.1
- XGBoost reg_lambda: 5
- XGBoost subsample: 0.7
- XGBoost colsample_bytree: 0.7
- CAGR: 2.17%
- Sharpe Ratio: 0.28
- Max Drawdown: 24.37%
- Win Rate: 60.25%
- Avg Excess Return: 0.62%
- Top 3 features: iv_rank (0.1135), atr_norm (0.0909), sma50_ratio (0.0856)
- Notes: 

## Run 7 — 2026-04-29
- Date range: 2009-01-01 to present
- Stock universe: 101 tickers
- Model: Ensemble (RF + XGBoost, 50/50 weighted average)
- Training Window: 252 days rolling
- Test Step: 5 days
- Top N Stocks: 5
- RF n_trees: 100
- RF n_subfeatures: 4
- RF max_depth: 10
- XGBoost num_round: 300
- XGBoost max_depth: 3
- XGBoost eta (learning rate): 0.03
- XGBoost min_child_weight: 3
- XGBoost gamma: 0.1
- XGBoost reg_lambda: 5
- XGBoost subsample: 0.7
- XGBoost colsample_bytree: 0.7
- CAGR: 2.11%
- Sharpe Ratio: 0.23
- Max Drawdown: 32.76%
- Win Rate: 58.32%
- Avg Excess Return: 0.58%
- Top 3 features: rsi (0.0847), sma20_ratio (0.0818), spy_corr (0.0805)
- Notes: 
