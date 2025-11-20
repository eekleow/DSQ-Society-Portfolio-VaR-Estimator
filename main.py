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
weights=dm.get_weights(TICKERS)
returns = data_train.pct_change().dropna()
cov_matrix=returns.cov().to_numpy()

optimal_weights, min_variance = sm.min_portfolio_variance(cov_matrix, len(TICKERS))
portfolio_optimised = pd.DataFrame({
    'ticker': TICKERS,
    'weight': optimal_weights
})
portfolio_optimised.to_csv("DATA/portfolio_optimised.csv", index=False)

# Backtesting