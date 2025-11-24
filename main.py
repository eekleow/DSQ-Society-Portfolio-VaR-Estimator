#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats
import datetime as dt
import requests
from io import StringIO
from scipy.optimize import minimize

from SCRIPTS import data_methods as dm
from SCRIPTS import stat_methods as sm

# Parameters
API_TOKEN = 'e2fd256de21fbae5eaf995d3e84002bcdcecfa85'
TICKERS = ['GLD', 'SPY', 'TSLA', 'BRK-B', 'AAPL']
START_DATE = '2023-01-01'
END_DATE = str(dt.date.today())
TEST_DATE = '2025-01-01'

# Data Retrieval
data = dm.get_data(TICKERS, START_DATE, END_DATE, API_TOKEN)

# Training and Optimisation
data_train = data[data.index < TEST_DATE]
weights_prior=dm.get_weights(TICKERS)
returns_train = data_train.pct_change().dropna()
cov_matrix=returns_train.cov().to_numpy()

weights_optimal, min_variance = sm.min_portfolio_variance(cov_matrix, len(TICKERS))
portfolio_optimised = pd.DataFrame({
    'ticker': TICKERS,
    'weight': weights_optimal.round(4)
})
portfolio_optimised.to_csv("DATA/portfolio_optimised.csv", index=False)

# Backtesting
returns = data.pct_change().dropna()
raw_exception_pct, raw_exception_ann_count = sm.parametric_backtest(returns, weights_prior)
opt_exception_pct, opt_exception_ann_count = sm.parametric_backtest(returns, weights_optimal)
backtest_metrics = pd.DataFrame({
    'Metric': ['Raw Portfolio Exception %', 'Raw Portfolio Annual Exception Count', 'Optimised Portfolio Exception %', 'Optimised Portfolio Annual Exception Count'],
    'Value': [round(raw_exception_pct,4)*100, raw_exception_ann_count, round(opt_exception_pct,4)*100, opt_exception_ann_count]
})
backtest_metrics.to_csv("DATA/backtest_metrics.csv", index=False)

#Plotting 
raw_img_path = "DATA/portfolio_raw_plot.png"
opt_img_path = "DATA/portfolio_optimised_plot.png"
raw_portfolio_plot = sm.varplots(returns_train,weights_prior, "raw",raw_img_path,TICKERS)
optimised_portfolio_plot = sm.varplots(returns_train,weights_optimal,"optimised",opt_img_path,TICKERS)
