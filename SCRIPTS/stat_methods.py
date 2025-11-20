#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from scipy.optimize import minimize

def portfolio_variance(weights,cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def min_portfolio_variance(cov_matrix, n):
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n
    optimise = minimize(portfolio_variance, init_guess, args=(cov_matrix,),method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12, 'eps': 1e-6})
    optimal_weights = optimise.x
    min_variance = optimise.fun
    return results