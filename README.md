# Portfolio VaR Risk Lab

A Python pipeline for **minimum-variance portfolio optimisation** and **multi-estimator Value-at-Risk (VaR)** estimation, with rolling-window backtesting and Basel-style traffic-light model validation.

## Overview

Given a set of equities and holding weights, this tool constructs the lowest-variance long-only portfolio, estimates 1-day downside risk under three distributional assumptions, and validates each VaR estimator against realised returns using a rolling backtest. It outputs CSV summary tables and annotated distribution plots.

## Key Features

- **Minimum-variance optimisation** — solves for portfolio weights that minimise `wᵀΣw` subject to full-investment and long-only constraints (SciPy SLSQP).
- **Three VaR estimators** — Historical (non-parametric empirical quantile), Parametric Normal, and Student-t (fat-tailed, fit via MLE).
- **Rolling-window backtest** — 2-year (504-day) rolling window with 1-day shift to prevent look-ahead bias; reports exception rate, annualised exception count, and traffic-light zone (Green / Yellow / Red).
- **Train/test discipline** — covariance is estimated on the training window only; VaR is evaluated out-of-sample on the test window.
- **Artifacts** — CSV summaries and PNG distribution plots (histogram + Normal/Student-t PDF overlays + VaR markers).

## Methodology

**Optimisation problem**
```
minimise   wᵀ Σ w
subject to Σ wᵢ = 1,   0 ≤ wᵢ ≤ 1   (long-only, fully invested)
```
Σ is the covariance matrix of training-window daily returns. Solved with `scipy.optimize.minimize` (SLSQP, `maxiter=1000`, `ftol=1e-12`).

**VaR estimators (1-day, 95% confidence / 5% left tail)**

| Method | Assumption | Formula |
|---|---|---|
| Historical | None (empirical) | 5th percentile of portfolio returns |
| Parametric Normal | Normal returns | `μ − 1.645·σ` |
| Student-t | Fat tails | `loc + t-quantile·scale`, params via `t.fit` (MLE) |

**Backtesting** — an exception occurs when the realised next-day return falls below the VaR estimate. Annualised exception counts map to zones: Green (0–4), Yellow (5–9), Red (≥10).

## Tech Stack

Python · NumPy · pandas · SciPy · SciPy.stats · Matplotlib · Tiingo API (with local CSV fallback)

## Project Structure

```
data_methods.py     # Tiingo API fetch + CSV fallback, weight loading
stat_methods.py     # Min-variance optimisation, VaR estimators, plots, backtests
run_var_report.py   # Orchestration: runs the pipeline, exports CSV + PNG
DATA/               # Backup price data
OUTPUT/             # Generated CSVs and plots
```

## How to Run

```bash
pip install -r requirements.txt
python run_var_report.py
```

Outputs are written to `OUTPUT/`:
- `weights.csv` — raw vs optimised weights
- `var_summary.csv` — VaR across all three estimators, both portfolios
- `var_report.csv` — exception rates, annual counts, zones
- `portfolio_raw_plot.png`, `portfolio_optimised_plot.png`

## Example Results

On a sample AAPL / WMT / GE portfolio, optimisation reallocated from equal weights toward AAPL (~63%), and backtesting flagged tail-risk underestimation in several estimators (Red zone), illustrating the practical limits of parametric VaR under fat-tailed returns.

## Assumptions & Limitations

- 1-day VaR on close-to-close returns; Normal VaR uses fixed z ≈ 1.645.
- Assumes the training-window covariance is informative for the future.
- VaR is **not subadditive** — covariance must be used rather than summing individual VaRs.
- Does not handle stale prices (assets must trade in the same time zone), low liquidity, or losses beyond the confidence interval (e.g. recession tail paths).
- Sensitive to lookback window and confidence-level choices.

## Real-World Applications

Portfolio risk limits and asset allocation, risk-adjusted performance measurement, and regulatory capital estimation (minimum reserve requirements).

## Authors

Atila, Royce, Terry, Ee-K
