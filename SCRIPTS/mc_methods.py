import pandas as pd
import numpy as np
import matplotlib as plt

def mcNormal(meanReturns, covMatrix, mc_sims):
    rng = np.random.default_rng() #did not pass a seed
    return rng.multivariate_normal(mean=meanReturns,cov=covMatrix,size=mc_sims)

'''
def mcStudentT(meanReturns, covMatrix, mc_sims, df=6, seed=None):
    """
    1-day Student-t Monte Carlo simulation.
    df must be > 2 for finite variance.
    """
    if df <= 2:
        raise ValueError("df must be > 2")

    rng = np.random.default_rng(seed)

    # dimensions
    N = len(meanReturns)

    # extract volatilities and correlation
    std = np.sqrt(np.diag(covMatrix))
    corr = covMatrix / np.outer(std, std)
    corr = (corr + corr.T) / 2  # numerical safety

    # Cholesky of correlation
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(corr + 1e-8 * np.eye(N))

    # 1) correlated normals
    Z = rng.standard_normal(size=(mc_sims, N)) @ L.T

    # 2) normal → uniform → student-t
    from scipy.stats import norm, t
    U = norm.cdf(Z)
    T = t.ppf(U, df=df)

    # 3) standardise variance to 1
    T = T / np.sqrt(df / (df - 2))

    # 4) scale + add mean
    sims = meanReturns + T * std
    return sims  # shape: (mc_sims, N)

'''
