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
from SCRIPTS import mc_methods as mm

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

#MC methods
returns_train = returns_train.mean()
mcN_asset_sims = mm.mcNormal(returns_train,cov_matrix,1000)
return_p = (mcN_asset_sims*weights_prior).sum(axis=1)
mcVaR=np.percentile(mcN_asset_sims, 5)
sns.histplot(return_p,bins=100)
print(mcVaR)
plt.show()

'''
portfolio_optimised.to_csv("DATA/portfolio_optimised.csv", index=False)

# Backtesting
data_test = data[data.index >= TEST_DATE]
returns_test = data_test.pct_change().dropna()
raw_historical_var, raw_parametric_var, raw_t_var = sm.calculate_var(returns_test.dot(weights_prior), confidence=0.95)
opt_historical_var, opt_parametric_var, opt_t_var = sm.calculate_var(returns_test.dot(weights_optimal), confidence=0.95)
# Parametric Backtesting
returns = data.pct_change().dropna()
raw_exception_pct, raw_exception_ann_count = sm.parametric_backtest(returns, weights_prior)
opt_exception_pct, opt_exception_ann_count = sm.parametric_backtest(returns, weights_optimal)
raw_zone = sm.classify_zone(raw_exception_ann_count)
opt_zone = sm.classify_zone(opt_exception_ann_count)
# T test backtesting 
t_raw_exception_pct, t_raw_exception_ann_count = sm.t_backtest(returns, weights_prior)
t_opt_exception_pct, t_opt_exception_ann_count = sm.t_backtest(returns, weights_optimal)
t_raw_zone = sm.classify_zone(t_raw_exception_ann_count)
t_opt_zone = sm.classify_zone(t_opt_exception_ann_count)

backtest_metrics = pd.DataFrame({
    'Metric': ['Raw Portfolio Historical VaR %',
               'Raw Portfolio Parametric VaR %',
               'Raw Portfollio T-Distribution VaR %',
               'Raw Portfolio Exception % under Parametric Distribution', 
               'Raw Portfolio Annual Exception Count under Parametric Distribution',
               'Raw Portfolio Zone under Parametric Distribution', 
               'Raw Portfolio Exception % under T-Distribution', 
               'Raw Portfolio Annual Exception Count under T-Distribution',
               'Raw Portfolio Zone under T-Distribution', 
               'Optimised Portfolio Historical VaR %',
               'Optimised Portfolio Parametric VaR %',
               'Optimised Portfollio T-Distribution VaR %',
               'Optimised Portfolio Exception %', 
               'Optimised Portfolio Annual Exception Count',
               'Optimised Portfolio Model Zone',
               'Optimised Portfolio Exception % under T-Distribution', 
               'Optimised Portfolio Annual Exception Count under T-Distribution',
               'Optimised Portfolio Zone under T-Distribution',
               ],
    'Value': [round(raw_historical_var,4)*100,
              round(raw_parametric_var,4)*100,
              round(raw_t_var,4)*100,
              round(raw_exception_pct,4)*100, 
              raw_exception_ann_count,
              raw_zone,
              round(t_raw_exception_pct,4)*100, 
              t_raw_exception_ann_count,
              t_raw_zone,
              round(opt_historical_var,4)*100,
              round(opt_parametric_var,4)*100,
              round(opt_t_var,4)*100,
              round(opt_exception_pct,4)*100, 
              opt_exception_ann_count,
              opt_zone,
              round(t_opt_exception_pct,4)*100, 
              t_opt_exception_ann_count,
              t_opt_zone
              ]
})
backtest_metrics.to_csv("DATA/backtest_metrics.csv", index=False)

#Plotting 
raw_img_path = "DATA/portfolio_raw_plot.png"
opt_img_path = "DATA/portfolio_optimised_plot.png"
raw_portfolio_plot = sm.varplots(returns_test,weights_prior, "raw",raw_img_path,TICKERS)
optimised_portfolio_plot = sm.varplots(returns_test,weights_optimal,"optimised",opt_img_path,TICKERS)
'''