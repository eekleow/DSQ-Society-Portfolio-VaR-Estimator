import numpy as np

def calculate_var(returns, confidence=0.95):
    # Historical VaR: just find the 5th percentile
    historical_var = np.percentile(returns, 5)
    
    # Parametric VaR: mean - 1.645*std (for 95% confidence)
    mean = returns.mean()
    std = returns.std()
    parametric_var = mean - 1.645 * std
    
    return historical_var, parametric_var