"""XGBoost NSE Trading Pipeline (1-minute or daily bars)

Modules:

- yahoo_minute_loader: Fetch 1-minute OHLCV from Yahoo Finance (--bars minute)
- csv_loader: Load daily OHLCV from the R analysis CSVs (--bars daily)
- data_loader: Alternative 1-minute loader for Angel One SmartAPI (needs a key; not used by main.py)
- fii_loader: Load daily FII/DII flows from the scraped CSV
- data_merger: Attach previous-day FII figures to each bar
- feature_engineer: Build ~30 technical + FII features
- label_creator: Create forward-return labels (next 5 minutes / next day)
- data_checks: Data-quality checks run between steps
- data_splitter: Time-safe train/test split + scaling
- model_trainer: XGBoost training with early stopping
- hyperparameter_tuner: Grid search & Bayesian optimization
- feature_importance: SHAP + gain analysis
- backtest_engine: Walk-forward backtesting
- results_reporter: Plots, CSVs, summary reports
"""
