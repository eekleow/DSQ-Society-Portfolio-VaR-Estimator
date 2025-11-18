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

from SCRIPTS import data_pull as dp
from SCRIPTS import var_calculations as vc

# Parameters
API_TOKEN = 'e2fd256de21fbae5eaf995d3e84002bcdcecfa85'
TICKERS = ['GLD', 'SPY', 'TSLA', 'BRK-B', 'AAPL']
START_DATE = '2023-01-01'
END_DATE = str(dt.date.today())

# Data Retrieval
data = dp.get_data(TICKERS, START_DATE, END_DATE, API_TOKEN)
weights=dp.get_weights(TICKERS)
log_returns = np.log(data/data.shift(1))
print(weights)
portfolio_returns = (returns * weights).sum(axis=1).dropna()

# VaR Calculations
historical_var, parametric_var = vc.calculate_var(portfolio_returns)
print(f"Historical VaR (95%): {historical_var}")
print(f"Parametric VaR (95%): {parametric_var}")
