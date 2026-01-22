#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns 
from scipy.stats import t


def portfolio_variance(weights,cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def min_portfolio_variance(cov_matrix, n):
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n
    optimise = minimize(portfolio_variance, init_guess, args=(cov_matrix,),method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-6})
    optimal_weights = optimise.x
    min_variance = optimise.fun
    return optimal_weights, min_variance

def parametric_backtest(returns, weights):
    portfolio_return=(returns*weights).sum(axis=1)
    portfolio_return.name = "return"
    portfolio_df = portfolio_return.to_frame()
    portfolio_df['rolling_mean_2y'] = portfolio_df["return"].rolling(window=504).mean()
    portfolio_df['rolling_std_2y'] = portfolio_df["return"].rolling(window=504).std()
    portfolio_df['parametric_var'] = (portfolio_df['rolling_mean_2y'] - 1.645 * portfolio_df['rolling_std_2y']).shift(1)
    portfolio_df = portfolio_df[["return", "parametric_var"]].dropna()
    portfolio_df['in_limit'] = portfolio_df['return'] < portfolio_df['parametric_var']
    count = portfolio_df['in_limit'].sum()
    percent=count/len(portfolio_df)
    ann_count = np.ceil(percent*252)
    return percent, ann_count

def t_backtest(returns, weights):
    portfolio_return=(returns*weights).sum(axis=1)
    portfolio_return.name = "return"
    portfolio_df = portfolio_return.to_frame()
    def fit_nu(x):
        nu, loc, scale = t.fit(x)
        return nu
    def fit_loc(x):
        nu, loc, scale = t.fit(x)
        return loc
    def fit_scale(x):
        nu, loc, scale = t.fit(x)
        return scale
    portfolio_df['nu']    =  portfolio_df['return'].rolling(window=504).apply(fit_nu, raw=False)
    portfolio_df['loc']   =  portfolio_df['return'].rolling(window=504).apply(fit_loc, raw=False)
    portfolio_df['scale'] = portfolio_df['return'].rolling(window=504).apply(fit_scale, raw=False)
    t_quantile = t.ppf(0.05,  portfolio_df['nu'])
    portfolio_df['t_var'] = (portfolio_df['loc'] + t_quantile * portfolio_df['scale']).shift(1)
    portfolio_df = portfolio_df[["return","t_var"]].dropna(subset=['t_var'])
    portfolio_df['in limit'] = portfolio_df['return'] < portfolio_df['t_var']
    count = portfolio_df['in limit'].sum()   
    percent=count/len(portfolio_df)
    ann_count = np.ceil(percent * 252)             
    return percent, ann_count

def classify_zone(x):
    if 0 <= x <= 4:
        return "Green Zone"
    elif 5 <= x <= 9:
        return "Yellow Zone"
    elif x >= 10:
        return "Red Zone"
    else:
        return ""

def varplots(returns,weights, weight_type, img_path,tickers):
    portfolio_returns = (returns * weights).sum(axis=1)
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    nu, loc, scale = t.fit(portfolio_returns)
    portfolio_parametric_var = mean - 1.645 * std
    portfolio_historic_var = np.percentile(portfolio_returns, 5)
    portfolio_t_var = loc + t.ppf(0.05,nu)*scale
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
    pdf = norm.pdf(x, mean, std)
    t_pdf = t.pdf(x, df=nu, loc=loc, scale=scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(portfolio_returns, bins=100, stat='density',label="Non-parametric Returns", ax=ax)
    ax.plot(x, pdf, linewidth=2, color='black', label='Parametric Returns')
    ax.plot(x, t_pdf, linewidth=2, color='blue', label='T-Distribution Returns')
    ax.axvline(portfolio_parametric_var, color='r', linestyle='dashed')
    ax.axvline(portfolio_historic_var, color='g', linestyle='dashed')
    ax.axvline(portfolio_t_var, color='blue', linestyle='dashed')
    ax.text(
        portfolio_parametric_var, ax.get_ylim()[1]*0.8,
        f"5% Parametric VaR = {portfolio_parametric_var:.5f}",
        color='r', ha='right', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    ax.text(
        portfolio_historic_var, ax.get_ylim()[1]*0.8,
        f"5% Historic VaR = {portfolio_historic_var:.5f}",
        color='g', ha='left', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    ax.text(
        portfolio_t_var, ax.get_ylim()[1]*0.7,
        f"5% T-Distribution VaR = {portfolio_t_var:.5f}",
        color='blue', ha='right', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    ax.text(
        0.98, 0.15,
        f"Mean = {mean:.5f}\nStd = {std:.5f}",
        transform=ax.transAxes,      
        ha='right', va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    df_weights = pd.DataFrame({"Ticker": tickers,"Weight": np.array(weights).round(2)})
    weights_text = df_weights.to_string(index=False)
    ax.text(
        0.98, 0.60,
        weights_text,
        transform=ax.transAxes,      
        ha='right', va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    ax.set_xlabel("Portfolio Percentage Return") 
    ax.set_title(f'Parametric, Non-parametric and T-Distribution YTD returns of '+ f', '.join(tickers) + f' using {weight_type} weights',wrap=True, pad=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)

def calculate_var(returns, confidence=0.95):
    historical_var = np.percentile(returns, 5)
    mean = returns.mean()
    std = returns.std()
    parametric_var = mean - 1.645 * std
    nu, loc, scale=t.fit(returns)
    t_var=loc + t.ppf(0.05, nu )* scale
    return historical_var, parametric_var, t_var
