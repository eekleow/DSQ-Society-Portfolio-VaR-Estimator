import os
import sys
import datetime as dt
import pandas as pd

from SCRIPTS import data_methods as dm
from SCRIPTS import stat_methods as sm


def ask(prompt: str, default: str | None = None) -> str:
    if default is None:
        return input(f"{prompt}: ").strip()
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def yesno(s: str) -> bool:
    return s.strip().lower() in {"y", "yes", "true", "1"}


def parse_date(s: str) -> str:
    try:
        dt.datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format (e.g., 2025-01-22).")


def parse_tickers(s: str) -> list[str]:
    tickers = [x.strip().upper() for x in s.split(",") if x.strip()]
    if not tickers:
        raise ValueError("Please provide at least 1 ticker.")
    return tickers


def parse_weights(weights_str: str, n: int) -> list[float]:
    """
    If weights_str empty -> equal weights.
    Else parse comma-separated floats, normalise to sum=1, validate length.
    """
    if not weights_str.strip():
        return [1.0 / n] * n

    parts = [p.strip() for p in weights_str.split(",") if p.strip()]
    weights = [float(p) for p in parts]

    if len(weights) != n:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of tickers ({n}).")

    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")

    s = sum(weights)
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")

    return [w / s for w in weights]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_tiingo_token(user_entry: str) -> str:
    """
    Priority:
    1) User typed it
    2) Env var TIINGO_API_TOKEN
    3) Env var API_TOKEN
    """
    if user_entry.strip():
        return user_entry.strip()

    token = os.environ.get("TIINGO_API_TOKEN", "").strip()
    if token:
        return token

    token = os.environ.get("API_TOKEN", "").strip()
    if token:
        return token

    raise ValueError(
        "No Tiingo API token provided.\n"
        "Either paste it when prompted, or set environment variable TIINGO_API_TOKEN."
    )


def main():
    print("\n=== VaR Risk Lab — Interactive Report Generator (Tiingo Live) ===\n")

    # Output folder (NEW)
    OUTPUT_DIR = "OUTPUT"
    ensure_dir(OUTPUT_DIR)

    tickers = parse_tickers(ask("Enter tickers (comma-separated)", "AAPL,NVDA,GOOG,LLY,WMT,GE"))
    start_date = parse_date(ask("Start date (YYYY-MM-DD)", "2023-01-01"))
    end_date = parse_date(ask("End date (YYYY-MM-DD)", str(dt.date.today())))
    test_date = parse_date(ask("Backtest start date / train-test split (YYYY-MM-DD)", "2025-01-01"))

    confidence_str = ask("VaR confidence level (e.g., 95 for 95%)", "95")
    try:
        confidence = float(confidence_str)
        if not (0 < confidence < 100):
            raise ValueError
    except ValueError:
        print("Confidence must be a number between 0 and 100 (exclusive).")
        sys.exit(1)

    run_optim = yesno(ask("Run min-variance optimisation? (y/n)", "y"))
    run_backtest = yesno(ask("Run backtests + zoning? (y/n)", "y"))

    weights_str = ask("Enter RAW weights (comma-separated) or press Enter for equal weights", "")
    weights_raw = parse_weights(weights_str, len(tickers))

    token_entry = ask("Tiingo API token (or leave blank to use env var TIINGO_API_TOKEN)", "")
    api_token = get_tiingo_token(token_entry)

    print("\n[1/5] Pulling LIVE data from Tiingo...")
    try:
        data = dm.get_data(tickers, start_date, end_date, api_token)
    except Exception as e:
        print("\n❌ Tiingo fetch failed.")
        print("Common causes: unsupported ticker symbol in Tiingo, rate limits, or network issues.")
        print(f"Details: {e}")
        sys.exit(1)

    if data.empty:
        print("\n❌ Tiingo returned empty data for the selected tickers/date range.")
        sys.exit(1)

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    returns = data.pct_change().dropna()

    print("[2/5] Preparing train/test split...")
    data_train = data[data.index < test_date]
    returns_train = data_train.pct_change().dropna()

    if returns_train.empty:
        print("\n❌ Training set is empty. Choose an earlier backtest start date (test_date).")
        sys.exit(1)

    cov = returns_train.cov().to_numpy()

    if run_optim:
        print("[3/5] Optimising min-variance weights (SLSQP, long-only, sum=1)...")
        weights_opt, _ = sm.min_portfolio_variance(cov, len(tickers))
    else:
        weights_opt = weights_raw

    print("[4/5] Computing VaR on test period...")
    data_test = data[data.index >= test_date]
    returns_test = data_test.pct_change().dropna()

    if returns_test.empty:
        print("\n❌ Test set is empty. Choose an earlier end date or earlier test_date.")
        sys.exit(1)

    raw_h, raw_p, raw_t = sm.calculate_var(returns_test.dot(weights_raw), confidence=confidence / 100)
    opt_h, opt_p, opt_t = sm.calculate_var(returns_test.dot(weights_opt), confidence=confidence / 100)

    print("[5/5] Generating distribution plots...")
    raw_img_path = os.path.join(OUTPUT_DIR, "portfolio_raw_plot.png")
    opt_img_path = os.path.join(OUTPUT_DIR, "portfolio_optimised_plot.png")
    sm.varplots(returns_test, weights_raw, "raw", raw_img_path, tickers)
    sm.varplots(returns_test, weights_opt, "optimised", opt_img_path, tickers)

    var_report_df = None
    if run_backtest:
        raw_exc_pct, raw_ann = sm.parametric_backtest(returns, weights_raw)
        opt_exc_pct, opt_ann = sm.parametric_backtest(returns, weights_opt)

        t_raw_exc_pct, t_raw_ann = sm.t_backtest(returns, weights_raw)
        t_opt_exc_pct, t_opt_ann = sm.t_backtest(returns, weights_opt)

        var_report_df = pd.DataFrame({
            "Model": ["Parametric", "T-dist", "Parametric", "T-dist"],
            "Portfolio": ["Raw", "Raw", "Optimised", "Optimised"],
            "Exception %": [raw_exc_pct, t_raw_exc_pct, opt_exc_pct, t_opt_exc_pct],
            "Annual exceptions": [raw_ann, t_raw_ann, opt_ann, t_opt_ann],
            "Zone": [
                sm.classify_zone(raw_ann),
                sm.classify_zone(t_raw_ann),
                sm.classify_zone(opt_ann),
                sm.classify_zone(t_opt_ann),
            ],
        })

    print(f"\nWriting CSV outputs to {OUTPUT_DIR}/ ...")

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Raw weight": weights_raw,
        "Optimised weight": weights_opt,
    })
    weights_csv = os.path.join(OUTPUT_DIR, "weights.csv")
    weights_df.to_csv(weights_csv, index=False)

    var_summary_df = pd.DataFrame({
        "Method": ["Historical (5%)", "Parametric Normal (5%)", "T-distribution (5%)"],
        "Raw VaR": [raw_h, raw_p, raw_t],
        "Optimised VaR": [opt_h, opt_p, opt_t],
    })
    var_summary_csv = os.path.join(OUTPUT_DIR, "var_summary.csv")
    var_summary_df.to_csv(var_summary_csv, index=False)

    var_report_csv = None
    if var_report_df is not None:
        var_report_csv = os.path.join(OUTPUT_DIR, "var_report.csv")
        var_report_df.to_csv(var_report_csv, index=False)

    print("✅ Outputs written:")
    print(f" - {weights_csv}")
    print(f" - {var_summary_csv}")
    if var_report_csv:
        print(f" - {var_report_csv}")

    print(f"Plots saved: {raw_img_path}, {opt_img_path}")
    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
