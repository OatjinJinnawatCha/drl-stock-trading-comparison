"""Summarize all seeds, including deteriorations after tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trading.config import ALGORITHMS, TICKERS, INITIAL_CAPITAL


def build_report(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    frozen = json.loads((directory / "selection_frozen.json").read_text())
    rows = pd.read_csv(directory / "test_per_seed.csv")
    fields = ["cumulative_return", "sharpe_ratio", "max_drawdown", "number_of_trades",
              "turnover", "total_fees", "final_value", "mean_position_ratio"]
    for field in fields:
        rows[field] = pd.to_numeric(rows[field], errors="raise")
    groups = rows.groupby(["ticker", "algorithm", "condition"])
    summary = groups[fields].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary["n_runs"] = groups.size()
    summary["n_defined_sharpe"] = groups["sharpe_ratio"].count()
    summary = summary.reset_index()
    summary.to_csv(directory / "test_summary.csv", index=False)
    paired = []
    for ticker in TICKERS:
        for algorithm in ALGORITHMS:
            subset = rows[(rows.ticker == ticker) & (rows.algorithm == algorithm)]
            default = subset[subset.condition == "Default"].set_index("seed")
            tuned = subset[subset.condition == "Tuned"].set_index("seed")
            assert default.index.equals(tuned.index)
            for seed in default.index:
                paired.append({"ticker": ticker, "algorithm": algorithm, "seed": int(seed),
                               **{f"{f}_delta": float(tuned.loc[seed, f] - default.loc[seed, f])
                                  for f in ("cumulative_return", "sharpe_ratio", "max_drawdown", "turnover")}})
    paired = pd.DataFrame(paired)
    paired.to_csv(directory / "paired_deltas.csv", index=False)
    paired_summary = paired.groupby(["ticker", "algorithm"]).agg(
        return_delta_mean=("cumulative_return_delta", "mean"),
        return_delta_std=("cumulative_return_delta", "std"),
        sharpe_delta_mean=("sharpe_ratio_delta", "mean"),
        drawdown_delta_mean=("max_drawdown_delta", "mean"),
        seed_win_fraction=("cumulative_return_delta", lambda s: (s > 0).mean()),
    ).reset_index()
    paired_summary.to_csv(directory / "paired_summary.csv", index=False)

    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    screen = pd.read_csv(directory / "screen_ranking.csv")
    confirm = pd.read_csv(directory / "confirm_ranking.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for algorithm, ax in zip(ALGORITHMS, axes):
        points = screen[screen.algorithm == algorithm].copy()
        labels = points.candidate.tolist()
        x = np.arange(len(points))
        ax.bar(x, points.validation_sharpe_mean, yerr=points.validation_sharpe_std,
               color="#acbccd", capsize=3, label="Screen (mean +/- stock/seed SD)")
        confirmed = confirm[confirm.algorithm == algorithm]
        for _, row in confirmed.iterrows():
            ax.scatter(labels.index(row.candidate), row.validation_sharpe_mean, marker="D",
                       color="#d57720", s=65, zorder=3)
        winner = frozen["winners"][algorithm]["id"]
        ax.set_title(f"{algorithm}\nSelected: {winner}")
        ax.set_xticks(x, labels, rotation=60, fontsize=8)
        ax.set_ylabel("Validation Sharpe")
        ax.axhline(0, color="black", linewidth=.6)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Validation-only search: bars = screen; orange diamonds = fresh-seed confirmation")
    fig.savefig(figures / "validation_search.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for i, ticker in enumerate(TICKERS):
        for j, algorithm in enumerate(ALGORITHMS):
            ax = axes[i, j]
            subset = rows[(rows.ticker == ticker) & (rows.algorithm == algorithm)]
            for role, color in (("Default", "#7b8794"), ("Tuned", "#156da8")):
                histories = [pd.read_csv(directory / path / "test_history.csv")
                             for path in subset[subset.condition == role].job_path]
                values = np.array([h.value.to_numpy() for h in histories])
                dates = pd.to_datetime(histories[0].date)
                mean = values.mean(0)
                ax.plot(dates, mean, label=role, color=color, linewidth=1.6)
                if len(values) > 1:
                    sd = values.std(0, ddof=1)
                    ax.fill_between(dates, mean - sd, mean + sd, color=color, alpha=.13)
            baseline = pd.read_csv(directory / "baselines" / f"{ticker}_Buy-and-Hold.csv")
            ax.plot(pd.to_datetime(baseline.date), baseline.value, linestyle="--", color="black",
                    label="Buy-and-Hold", alpha=.6, linewidth=1)
            ax.axhline(INITIAL_CAPITAL, color="black", linewidth=.5, alpha=.2)
            ax.set_title(f"{ticker} / {algorithm}")
            ax.set_ylabel("Portfolio value ($)")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    fig.suptitle("Test 2024-2025: paired fresh seeds (mean +/- sample SD)")
    fig.savefig(figures / "default_vs_tuned.png", dpi=150)
    plt.close(fig)

    table = ["| Stock | Algorithm | Condition | Return mean ± SD | Sharpe mean | Max DD mean | Trades mean |",
             "|---|---|---|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        ret = f"{r.cumulative_return_mean:.2%}"
        if np.isfinite(r.cumulative_return_std):
            ret += f" ± {r.cumulative_return_std:.2%}"
        sharp = f"{r.sharpe_ratio_mean:.3f}" if np.isfinite(r.sharpe_ratio_mean) else "NA"
        table.append(f"| {r.ticker} | {r.algorithm} | {r.condition} | {ret} | {sharp} | "
                     f"{r.max_drawdown_mean:.2%} | {r.number_of_trades_mean:.1f} |")
    delta_table = ["| Stock | Algorithm | Mean return change (percentage points) | Seed wins | Mean Sharpe change |",
                   "|---|---|---:|---:|---:|"]
    for _, r in paired_summary.iterrows():
        delta_table.append(f"| {r.ticker} | {r.algorithm} | {100*r.return_delta_mean:+.2f} | "
                           f"{r.seed_win_fraction:.0%} | {r.sharpe_delta_mean:+.3f} |")
    params = "\n\n".join(f"### {algorithm}\n\nSelected `{w['id']}`; confirmation validation Sharpe "
                              f"{w['validation_sharpe_mean']}.\n\n```json\n"
                              + json.dumps(w["params"], indent=2) + "\n```"
                              for algorithm, w in frozen["winners"].items())
    text = f"""# Hyperparameter comparison for concept 2

Mode: **{protocol['mode']}**. Best means best among the tested conditions on
2022–2023 validation data, not a globally optimal trading configuration.

Search used {len(next(iter(protocol['candidates'].values())))} parameter sets per
algorithm including v6 defaults, {protocol['screen_steps']:,} steps per screening
job, and {protocol['confirm_and_final_steps']:,} steps per confirmation/final job.
Screen seeds: {protocol['screen_seeds']}; confirmation seeds: {protocol['confirm_seeds']};
final paired seeds: {protocol['final_seeds']}. Each condition is evaluated on both
AMD and OXY with equal stock weight. One common winner is selected per algorithm.

Primary criterion: mean validation Sharpe across stock/seed cells. Every cell
must have defined Sharpe. Ties use higher return, higher (less negative) drawdown,
then candidate ID. Checkpoints use validation Sharpe with earliest tie-breaking.
Top conditions are retrained from scratch with fresh confirmation seeds; these
results alone select the final condition. Selection was frozen at
{frozen['frozen_at_utc']} before any v7 test evaluation.

**The same 2024–2025 test was already reported in v6.** This comparison is an
exploratory reuse of that historical period, not a newly untouched holdout. No
v7 test result is used to expand the search or change the selected parameters.

## Selected conditions

{params}

## Validation search

![Validation search](figures/validation_search.png)

The screening stage uses fewer steps; early rankings can miss slower learners.
Bars show variability across stock/seed cells, not uncertainty over future markets.
The orange points use different seeds and a larger training budget, so their
differences from the screening bars are not pure seed effects.

## Final paired-seed test

""" + "\n".join(table) + "\n\n## Change from matched defaults\n\n" + "\n".join(delta_table) + """

Positive return change means tuning increased mean historical test return.
Win fraction counts strictly positive paired return differences. If the default
condition itself wins, identical jobs are reused under both labels; zero change
is reported. The default is not excluded to manufacture an improvement.

![Default versus tuned](figures/default_vs_tuned.png)

## Controlled factors and limitations

The frozen v6 data snapshot, train-only scaling, ten state features, trading ledger,
Sell50/Hold/Buy50 actions, reward, 0.1% fees and $10,000 starting cash are unchanged.
Training remains 2015–2021; validation remains 2022–2023; test remains 2024–2025.
SAC remains the explicitly named SAC-Discrete variant. Final default and tuned
conditions receive the same training budget, checkpoint opportunities and seeds.

This is a small, budgeted categorical random search, not an exhaustive grid or
proof of optimality. Multiple parameters vary together, so these comparisons
do not isolate causal effects of individual hyperparameters. Five paired final
seeds describe training variability on one market path; they do not establish
statistical or economic superiority on future data. A better validation score
can coincide with worse test performance; all such results are retained.
Undefined Sharpe is reported as NA; defined-Sharpe counts appear in test_summary.csv.

No deposits, borrowing, shorting, cash interest, slippage, market impact, taxes
or terminal forced liquidation are modeled. Adjusted OHLC is a total-return
proxy. Baselines retain the same execution/accounting rules as v6.

Raw evidence: protocol.json, screen_ranking.csv, confirm_ranking.csv,
selection_frozen.json, test_per_seed.csv, paired_deltas.csv, every checkpoint,
validation curve and daily ledger. Parameter rationale and primary sources:
../../HYPERPARAMETER_STUDY.md. The v6 results remain unmodified.
"""
    if protocol["mode"] == "smoke":
        text += ("\n**SMOKE ONLY: these short integration runs are not research findings. "
                 "If every condition has undefined Sharpe, a fixed condition is used only "
                 "to exercise final-stage plumbing. The full study never uses this fallback.**\n")
    (directory / "REPORT.md").write_text(text)
