import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import TICKERS, INITIAL_CAPITAL


def build_report(run_dir):
    summary = pd.read_csv(run_dir / "summary.csv")
    config = json.loads((run_dir / "config.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for ticker, ax in zip(TICKERS, axes):
        for method in summary["method"].unique():
            files = sorted((run_dir / "histories").glob(f"{ticker}_{method}_*.csv"))
            histories = [pd.read_csv(p) for p in files]
            curves = np.array([np.r_[INITIAL_CAPITAL, h["value"].to_numpy()] for h in histories])
            dates = pd.to_datetime(histories[0]["date"])
            # Keep the pre-trade initial value in the plotted path and drawdown.
            dates = pd.DatetimeIndex([dates.iloc[0] - pd.offsets.BDay(1), *dates])
            mean = curves.mean(axis=0)
            line, = ax.plot(dates, mean, label=method, linewidth=1.5)
            if len(curves) > 1:
                std = curves.std(axis=0, ddof=1)
                ax.fill_between(dates, mean - std, mean + std, color=line.get_color(), alpha=0.1)
        ax.set_title(f"{ticker}: held-out test 2024-2025 (mean +/- sample SD across seeds)")
        ax.set_ylabel("Portfolio value ($)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8)
    fig.savefig(run_dir / "figures" / "equity_comparison.png", dpi=150)
    plt.close(fig)
    table = ["| Stock | Method | Return mean ± SD | Sharpe mean | Max DD mean | Trades mean | Turnover mean |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        sd = row["cumulative_return_std"]
        ret = f"{row['cumulative_return_mean']:.2%}"
        if np.isfinite(sd):
            ret += f" ± {sd:.2%}"
        table.append(f"| {row['ticker']} | {row['method']} | {ret} | {row['sharpe_ratio_mean']:.3f} | "
                     f"{row['max_drawdown_mean']:.2%} | {row['number_of_trades_mean']:.1f} | {row['turnover_mean']:.2f} |")
    text = f"""# Comparative DRL stock trading experiment

Mode: **{config['mode']}**. Actual Yahoo Finance data; no synthetic prices.
Agents: Dueling DDQN, PPO, and SAC-Discrete. Assets: AMD and OXY, separately trained.
Budget: {config['training_steps_per_run']:,} environment transitions per agent/stock/seed.
Seeds: {config['seeds']}. Baselines are deterministic and run once per stock.
Train 2015–2021; validation 2022–2023; test 2024–2025.
Checkpoint selected by validation Sharpe; no selection using test results.

""" + "\n".join(table) + """

Return is final value / initial capital - 1. SD uses ddof=1 for all agents.
Sharpe uses annualized daily excess returns over lagged ^IRX yields.
Sharpe is undefined (NA) for a portfolio with zero daily return variability.
Max drawdown includes initial capital before the first trade. Turnover is the sum
of traded notional / prior-day portfolio value (one-way, not divided by two).
Trades count executed orders above $0.000001; no-op commands are excluded.

![Equity comparison](figures/equity_comparison.png)

## Interpretation and limits

This run compares algorithm families under one fixed training budget and one
held-out market period. Equal environment steps do not imply equal gradient
updates or compute. Seed variability is training variability, not a confidence
interval for future market returns. No claim of algorithm superiority follows
from a single test period or only two assets. No test-driven tuning is performed.

SAC-Discrete is the categorical-action SAC variant, not continuous SB3 SAC.
All three agents execute the same Sell50/Hold/Buy50 actions. Baselines use their
own deterministic allocation rules through the same cash/share ledger, market
prices, dates, fees and starting capital. DCA spends the original $10,000 in 20
equal installments on evenly spaced trading days, with no outside deposits.

Features observed at the previous close; orders execute at the next open and
portfolios are marked to that day's close. Market indicators are train-scaled;
cash/position fractions are left as fractions. No borrowing, shorting, slippage,
cash interest, or terminal forced sale. Final stock holdings are marked to market.
Adjusted OHLC provides a total-return proxy including corporate actions, and is
not a reconstruction of historical unadjusted share counts or dividend cash flows.

See config.json, per-seed metrics, daily ledgers, validation checkpoint scores,
saved weights, train scalers, and data/source_manifest.json for audit evidence.
"""
    if config["mode"] == "smoke":
        text += "\n**Smoke test only: this budget is for integration validation, not research conclusions.**\n"
    (run_dir / "REPORT.md").write_text(text)
