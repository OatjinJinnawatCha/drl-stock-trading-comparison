"""Phase 7 tables and figures, read-only over results/phase7. Safe to rerun.

Bootstrap: paired moving-block (20 days, 5,000 draws) on daily returns of the equal-weight
average of the 5 seed portfolios minus Buy-and-Hold, per EXPERIMENT_PLAN.md section 6.
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading.config import ROOT, TICKERS, ALGORITHMS, SEEDS, TARGET_WEIGHTS

OUT = ROOT / "results" / "phase7"
FIG = OUT / "figures"
AGENTS = {"DuelingDDQN": "Dueling DDQN", "PPO": "PPO", "SAC-Discrete": "Discrete SAC"}
COLOR = {"DuelingDDQN": "#2a78d6", "PPO": "#eb6834", "SAC-Discrete": "#1baf7a"}
BASE = {"Buy-and-Hold": ("#0b0b0b", "--"), "SMA-Crossover": ("#52514e", ":"), "DCA": ("#8a8984", "-.")}
BUY, SELL = "#2a78d6", "#eb6834"
METRICS = ["cumulative_return", "sharpe_ratio", "max_drawdown", "number_of_trades", "turnover",
           "exposure", "mean_position_ratio", "total_fees"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e8e7e3",
                     "grid.linewidth": .6, "axes.axisbelow": True, "legend.frameon": False,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 300,
                     "figure.dpi": 150, "lines.linewidth": 1.5})
FULL = 6.4


def date_axis(ax, period):
    months = (1, 7) if period == "test" else (1, 4, 7, 10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def history(period, ticker, name):
    frame = pd.read_csv(OUT / period / "histories" / f"{ticker}_{name}.csv", parse_dates=["date"])
    return frame.set_index("date")


def agent_values(period, ticker, algorithm):
    return pd.DataFrame({s: history(period, ticker, f"{algorithm}_{s}")["value"] for s in SEEDS})


def sharpe(returns, rf):
    excess = returns - rf
    return np.sqrt(252) * excess.mean(-1) / excess.std(-1, ddof=1)


def block_bootstrap(a, b, rf, rng, block=20, draws=5000):
    n = len(a)
    starts = rng.integers(0, n, (draws, n // block + 1))
    index = ((starts[..., None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    return sharpe(a[index], rf[index]) - sharpe(b[index], rf[index])


def tables(frame):
    summary = frame.groupby(["period", "ticker", "method", "kind"], sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(OUT / "summary.csv", index=False)
    rows = []
    rng = np.random.default_rng(631)
    for period in ("test", "transfer_2026"):
        for ticker in TICKERS:
            base = history(period, ticker, "Buy-and-Hold")
            base_values = np.r_[10000.0, base["value"].to_numpy()]
            base_r = base_values[1:] / base_values[:-1] - 1
            rf = (1 + base["risk_free_annual"].to_numpy()) ** (1 / 252) - 1
            for algorithm in ALGORITHMS:
                values = np.r_[10000.0, agent_values(period, ticker, algorithm).mean(axis=1).to_numpy()]
                returns = values[1:] / values[:-1] - 1
                diffs = block_bootstrap(returns, base_r, rf, rng)
                low, high = np.percentile(diffs, [2.5, 97.5])
                rows.append({"period": period, "ticker": ticker, "method": algorithm,
                             "pooled_sharpe": float(sharpe(returns, rf)),
                             "buy_hold_sharpe": float(sharpe(base_r, rf)),
                             "difference": float(sharpe(returns, rf) - sharpe(base_r, rf)),
                             "ci_low": float(low), "ci_high": float(high),
                             "share_of_draws_above_zero": float((diffs > 0).mean())})
    boot = pd.DataFrame(rows)
    boot.to_csv(OUT / "bootstrap_sharpe_vs_buy_hold.csv", index=False)
    return boot


def fig_equity(period, name):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7), sharex=True)
    for ax, ticker in zip(axes, TICKERS):
        for algorithm in ALGORITHMS:
            values = agent_values(period, ticker, algorithm) / 1000
            ax.fill_between(values.index, values.min(axis=1), values.max(axis=1), color=COLOR[algorithm], alpha=.15, lw=0)
            ax.plot(values.index, values.mean(axis=1), color=COLOR[algorithm], label=AGENTS[algorithm])
        base = history(period, ticker, "Buy-and-Hold")["value"] / 1000
        ax.plot(base.index, base, color=BASE["Buy-and-Hold"][0], ls="--", lw=1.2, label="Buy-and-Hold")
        ax.axhline(10, color="#8a8984", lw=.6)
        ax.set_title(ticker, loc="left", fontweight="bold")
        date_axis(ax, period)
    axes[0].set_ylabel("Portfolio value ($ thousand)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_drawdown(period, name):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7), sharex=True, sharey=True)
    for ax, ticker in zip(axes, TICKERS):
        for strategy, (color, style) in BASE.items():
            value = history(period, ticker, strategy)["value"]
            ax.plot(value.index, 100 * (value / value.cummax() - 1), color=color, ls=style, lw=1.1, label=strategy)
        for algorithm in ALGORITHMS:
            value = agent_values(period, ticker, algorithm).mean(axis=1)
            ax.plot(value.index, 100 * (value / value.cummax() - 1), color=COLOR[algorithm], label=AGENTS[algorithm])
        ax.set_title(ticker, loc="left", fontweight="bold")
        date_axis(ax, period)
    axes[0].set_ylabel("Drawdown (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(.5, -.03), columnspacing=1, handlelength=1.8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_seed_sharpe(frame, period, name):
    data = frame[frame.period == period]
    methods = [*BASE, *ALGORITHMS]
    labels = [*BASE, *AGENTS.values()]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True, sharex=True)
    for ax, ticker in zip(axes, TICKERS):
        for row, method in enumerate(reversed(methods)):
            values = data[(data.ticker == ticker) & (data.method == method)].sharpe_ratio.to_numpy()
            if method in BASE:
                ax.scatter(values, [row], marker="D", s=26, color=BASE[method][0], zorder=3)
            else:
                ax.scatter(values, [row] * len(values), s=20, color=COLOR[method], edgecolor="white", lw=.5, zorder=3)
                ax.plot([values.mean()] * 2, [row - .3, row + .3], color="#0b0b0b", lw=1.5, zorder=4)
        ax.axvline(0, color="#8a8984", lw=.6)
        ax.set_yticks(range(len(methods)), list(reversed(labels)))
        ax.set_title(ticker, loc="left", fontweight="bold")
        ax.set_xlabel("Sharpe ratio")
        ax.grid(axis="y", visible=False)
    fig.text(.5, -.03, "Dots: individual seeds. Vertical bar: mean of 5 seeds. Diamonds: deterministic baselines.",
             ha="center", color="#52514e", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def action_shares(period):
    rows = {}
    for ticker in TICKERS:
        for algorithm in ALGORITHMS:
            actions = pd.concat([history(period, ticker, f"{algorithm}_{s}")["action"] for s in SEEDS])
            rows[(ticker, algorithm)] = actions.value_counts(normalize=True).reindex(range(5), fill_value=0)
    return pd.DataFrame(rows).T


def fig_actions(period, name):
    shares = action_shares(period)
    shares.to_csv(OUT / f"action_shares_{period}.csv")
    ramp = ["#cfe2f8", "#9cc4ef", "#5e9fe3", "#2a78d6", "#134a8c"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.7), sharey=True)
    for ax, ticker in zip(axes, TICKERS):
        for row, algorithm in enumerate(reversed(ALGORITHMS)):
            left = 0.0
            for action in range(5):
                width = shares.loc[(ticker, algorithm), action]
                ax.barh(row, width - .004 if width > .004 else width, left=left, color=ramp[action], height=.62,
                        label=f"{TARGET_WEIGHTS[action]:.0%}" if row == 0 else None)
                left += width
        ax.set_yticks(range(3), [AGENTS[a] for a in reversed(ALGORITHMS)])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.set_title(ticker, loc="left", fontweight="bold")
        ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Target stock weight (share of trading days, all 5 seeds)", loc="lower center",
               ncol=5, bbox_to_anchor=(.5, -.2))
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def median_seed(frame, ticker, algorithm):
    data = frame[(frame.period == "test") & (frame.ticker == ticker) & (frame.method == algorithm)]
    return int(data.sort_values("cumulative_return").iloc[len(data) // 2].seed)


def fig_overlay(frame, name):
    fig, axes = plt.subplots(3, 2, figsize=(FULL, 5.4), sharex=True)
    chosen = {}
    for column, ticker in enumerate(TICKERS):
        price = pd.read_csv(ROOT / "data" / f"{ticker}.csv", parse_dates=["date"]).set_index("date")["Open"]
        for row, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, column]
            seed = median_seed(frame, ticker, algorithm)
            chosen[f"{ticker}_{algorithm}"] = seed
            data = history("test", ticker, f"{algorithm}_{seed}")
            change = data["shares"].diff().fillna(data["shares"])
            traded = data["traded"].astype(bool)
            series = price.reindex(data.index)
            ax.plot(series.index, series, color="#52514e", lw=.9)
            for mask, color, marker, label in ((traded & (change > 0), BUY, "^", "Buy"), (traded & (change < 0), SELL, "v", "Sell")):
                ax.scatter(series.index[mask], series[mask], color=color, marker=marker, s=9, lw=0, zorder=3, label=label)
            ax.set_title(f"{ticker} · {AGENTS[algorithm]} (seed {seed})", loc="left", fontsize=8)
            date_axis(ax, "test")
    for row in range(3):
        axes[row, 0].set_ylabel("Open price ($)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    (OUT / "overlay_seeds.json").write_text(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    table = pd.read_csv(OUT / "per_seed.csv")
    boot = tables(table)
    fig_equity("test", "fig1_equity_test.png")
    fig_drawdown("test", "fig2_drawdown_test.png")
    fig_seed_sharpe(table, "test", "fig3_seed_sharpe_test.png")
    fig_actions("test", "fig4_actions_test.png")
    fig_overlay(table, "fig5_trading_overlay_test.png")
    fig_equity("transfer_2026", "fig6_equity_2026.png")
    fig_seed_sharpe(table, "transfer_2026", "fig7_seed_sharpe_2026.png")
    print(boot.round(3).to_string())
    fees = table[table.kind == "agent"].groupby(["period", "ticker", "method"], sort=False).total_fees.mean().round(0)
    print(fees.to_string())
    print(action_shares("test").round(2).to_string())
