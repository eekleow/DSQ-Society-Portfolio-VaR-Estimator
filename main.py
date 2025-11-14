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

from SCRIPTS import data_pull as dp

# Parameters
API_TOKEN = 'e2fd256de21fbae5eaf995d3e84002bcdcecfa85'
TICKERS = ['GLD', 'SPY', 'TSLA', 'BRK-B', 'AAPL']
START_DATE = '2023-01-01'
END_DATE = str(dt.date.today())

# Data Retrieval
data = dp.get_data(TICKERS, START_DATE, END_DATE, API_TOKEN)
weights=dp.get_weights(TICKERS)
returns = data.pct_change().dropna()
cov_matrix=returns.cov().to_numpy()

def portfolio_variance(weights,cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

n = len(TICKERS)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(n))
init_guess = np.ones(n) / n
optimise = minimize(portfolio_variance, init_guess, args=(cov_matrix),method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights=optimise.x
min_variance=optimise.fun