#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns 


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

def varplots(returns,weights, weight_type, img_path,tickers):
    portfolio_returns = (returns * weights).sum(axis=1)
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    portfolio_var = mean - 1.645 * std
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
    pdf = norm.pdf(x, mean, std)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(portfolio_returns, bins=100, stat='density',label="Non-parametric Returns", ax=ax)
    ax.plot(x, pdf, linewidth=2, color='green', label='Parametric Returns')
    ax.axvline(portfolio_var, color='r', linestyle='dashed')
    ax.text(
        portfolio_var, ax.get_ylim()[1]*0.8,
        f"5% VaR = {portfolio_var:.5f}",
        color='r', ha='right', fontsize=10,
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
    ax.set_xlabel("Portfolio Return") 
    ax.set_title(f'Parametric and Non-parametric returns of '+ f', '.join(tickers) + f' using {weight_type} weights',wrap=True, pad=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)

def calculate_var(returns, confidence=0.95):
    # Historical VaR: just find the 5th percentile
    historical_var = np.percentile(returns, 5)
    
    # Parametric VaR: mean - 1.645*std (for 95% confidence)
    mean = returns.mean()
    std = returns.std()
    parametric_var = mean - 1.645 * std
    
    return historical_var, parametric_var
