"""Post-hoc, exploratory explanation of the Phase 7 results. Does not load or change any model.
Uses recorded action histories only; the zero-fee replay re-prices the same action sequence.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading.config import ROOT, TICKERS, ALGORITHMS, SEEDS, SPLITS, MARKET_FEATURES, TARGET_WEIGHTS
from trading.data import prepare_data, TrainScaler
from trading.environment import TradingEnv
from trading.evaluation import metrics

OUT = ROOT / "results" / "phase7"
DIAG = OUT / "diagnostics"
DIAG.mkdir(exist_ok=True)
frames = prepare_data()
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)


def spearman(x, y):
    """Rank correlation and a two-sided normal-approximation p-value."""
    r = pd.Series(np.asarray(x)).rank().corr(pd.Series(np.asarray(y)).rank())
    return r, math.erfc(abs(r) * (len(x) - 1) ** .5 / 2 ** .5)


def hist(ticker, name, period="test"):
    return pd.read_csv(OUT / period / "histories" / f"{ticker}_{name}.csv", parse_dates=["date"]).set_index("date")


# A. Market regime per split
rows = []
for t in TICKERS:
    for split, (a, b) in SPLITS.items():
        f = frames[t].loc[a:b]
        r = f["Close"].pct_change().dropna()
        v = f["Close"]
        rows.append({"ticker": t, "split": split, "days": len(f), "buy_hold": v.iloc[-1] / v.iloc[0] - 1,
                     "ann_return": (v.iloc[-1] / v.iloc[0]) ** (252 / len(f)) - 1, "ann_vol": r.std() * 252 ** .5,
                     "daily_std": r.std(), "mdd": (v / v.cummax() - 1).min(), "lag1_autocorr": r.autocorr(1),
                     "sma10_above_30": (f["sma10"] > f["sma30"]).mean(),
                     "up_days": (r > 0).mean()})
regime = pd.DataFrame(rows); regime.to_csv(DIAG / "A_regime.csv", index=False)
print("A. REGIME\n", regime.round(3).to_string(index=False))

# B. Does any market feature predict the return actually earned after the decision?
rows = []
for t in TICKERS:
    f = frames[t]
    forward = f["Open"].shift(-1) / f["Open"] - 1   # earned from open t to open t+1
    for feature in MARKET_FEATURES:
        signal = f[feature].shift(1)                # observed at close t-1
        row = {"ticker": t, "feature": feature}
        for split, (a, b) in SPLITS.items():
            x = pd.concat([signal, forward], axis=1).loc[a:b].dropna()
            row[split] = spearman(x.iloc[:, 0], x.iloc[:, 1])[0]
        rows.append(row)
ic = pd.DataFrame(rows); ic.to_csv(DIAG / "B_feature_ic.csv", index=False)
print("\nB. RANK IC feature(t-1) vs return open(t)->open(t+1); |IC| noise band ~", round(1.96 / 1762 ** .5, 3), "(train)", round(1.96 / 502 ** .5, 3), "(test)")
print(ic.round(3).to_string(index=False))
print("train/test same sign:", int((np.sign(ic.train) == np.sign(ic.test)).sum()), "of", len(ic),
      "| train/validation same sign:", int((np.sign(ic.train) == np.sign(ic.validation)).sum()),
      "| corr(train IC, test IC):", round(ic.train.corr(ic.test), 2))

# C. Does Validation Sharpe predict Test Sharpe?
table = pd.read_csv(OUT / "per_seed.csv")
records = [json.loads(p.read_text()) for p in (ROOT / "results/phase6/final").glob("*/*/*/seed_*/complete.json")]
val = pd.DataFrame([{"ticker": r["ticker"], "method": r["algorithm"], "seed": r["seed"], "val_sharpe": r["validation"]["sharpe_ratio"],
                     "selected_step": r["selected_step"]} for r in records])
test = table[(table.period == "test") & (table.kind == "agent")].merge(val, on=["ticker", "method", "seed"])
y2026 = table[(table.period == "transfer_2026") & (table.kind == "agent")][["ticker", "method", "seed", "sharpe_ratio"]].rename(columns={"sharpe_ratio": "s26"})
test = test.merge(y2026, on=["ticker", "method", "seed"])
test["vd"] = test.val_sharpe - test.groupby(["ticker", "method"]).val_sharpe.transform("mean")
test["td"] = test.sharpe_ratio - test.groupby(["ticker", "method"]).sharpe_ratio.transform("mean")
test["sd"] = test.s26 - test.groupby(["ticker", "method"]).s26.transform("mean")
print("\nC. VALIDATION -> TEST across 30 runs")
print(" spearman val vs test, within-pair demeaned:", np.round(spearman(test.vd, test.td), 3))
print(" spearman test vs 2026, within-pair demeaned:", np.round(spearman(test.td, test.sd), 3))
print(" mean validation Sharpe", round(test.val_sharpe.mean(), 3), " mean test Sharpe", round(test.sharpe_ratio.mean(), 3))
print(" spearman selected_step vs test (demeaned):", np.round(spearman(test.selected_step, test.td), 3))
print(test.groupby("ticker")[["val_sharpe", "sharpe_ratio"]].mean().round(2).to_string())

# D-F. Decomposition per agent on Test
main = {t: TrainScaler().fit(frames[t]).transform(frames[t]) for t in TICKERS}
rows, churn, drivers = [], [], []
for t in TICKERS:
    bh = hist(t, "Buy-and-Hold")
    f = frames[t].loc[bh.index]
    stock = np.r_[bh["value"].iloc[0] / 10000 - 1, bh["value"].pct_change().dropna().to_numpy()]
    fwd = (frames[t]["Open"].shift(-1) / frames[t]["Open"] - 1).loc[bh.index]
    daily_std = f["Close"].pct_change().std()
    x_all = pd.DataFrame(main[t], index=frames[t].index, columns=MARKET_FEATURES).shift(1).loc[bh.index]
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            h = hist(t, f"{algorithm}_{seed}")
            env = TradingEnv(frames[t], main[t], "test", transaction_cost=0)
            env.reset()
            infos = [env.step(a)[4] for a in h["action"].to_numpy()]
            gross = metrics(pd.DataFrame(infos))
            net = metrics(h.reset_index())
            w = h["action"].map(lambda a: TARGET_WEIGHTS[a]).to_numpy()
            mean_w = h["position_ratio"].mean()
            # Static benchmark: buy once at the agent's own average weight, then never trade.
            stock_path = bh["value"].to_numpy() / 10000
            static = mean_w * stock_path + (1 - mean_w)
            s_r = np.diff(np.r_[1.0, static])
            s_m = {"ret": static[-1] - 1, "mdd": (static / np.maximum.accumulate(static) - 1).min()}
            agent_r = np.r_[h["value"].iloc[0] / 10000 - 1, h["value"].pct_change().dropna().to_numpy()]
            dw = np.abs(np.diff(w))
            rows.append({"ticker": t, "method": algorithm, "seed": seed, "net": net["cumulative_return"], "gross": gross["cumulative_return"],
                         "net_sharpe": net["sharpe_ratio"], "gross_sharpe": gross["sharpe_ratio"], "fees": net["total_fees"],
                         "mean_w": mean_w, "static_ret": s_m["ret"], "static_mdd": s_m["mdd"], "mdd": net["max_drawdown"],
                         "timing_corr": np.corrcoef(w[:-1], fwd.to_numpy()[:-1])[0, 1], "beta": np.polyfit(stock, agent_r, 1)[0]})
            churn.append({"ticker": t, "method": algorithm, "seed": seed, "target_changed": (dw > 0).mean(), "mean_abs_dw": dw.mean(),
                          "jump_ge_50": (dw >= .5).mean(), "trade_days": h["traded"].mean(),
                          "mean_run_days": len(w) / (1 + (dw > 0).sum()), "fee_bps_per_day": 1e4 * h["fee"].sum() / h["value"].mean() / len(h),
                          "reversal_next_day": np.mean(np.sign(np.diff(w))[1:] * np.sign(np.diff(w))[:-1] < 0) })
            X = np.c_[np.ones(len(x_all)), x_all.to_numpy()]
            coef, *_ = np.linalg.lstsq(X, w, rcond=None)
            pred = X @ coef
            drivers.append({"ticker": t, "method": algorithm, "seed": seed, "r2": 1 - ((w - pred) ** 2).sum() / ((w - w.mean()) ** 2).sum(),
                            **dict(zip(MARKET_FEATURES, coef[1:]))})
decomp = pd.DataFrame(rows); decomp.to_csv(DIAG / "D_decomposition.csv", index=False)
print("\nD. DECOMPOSITION (mean of 5 seeds), Test. Buy-and-Hold net:", {t: round(hist(t, 'Buy-and-Hold')['value'].iloc[-1] / 10000 - 1, 3) for t in TICKERS})
d = decomp.groupby(["ticker", "method"], sort=False).mean(numeric_only=True).drop(columns="seed")
print(d.round(3).to_string())
print("seeds beating own static mix, net:", decomp.assign(w=decomp.net > decomp.static_ret).groupby(["ticker", "method"], sort=False).w.sum().to_dict())
print("seeds beating own static mix, gross:", decomp.assign(w=decomp.gross > decomp.static_ret).groupby(["ticker", "method"], sort=False).w.sum().to_dict())
c = pd.DataFrame(churn); c.to_csv(DIAG / "E_churn.csv", index=False)
print("\nE. CHURN\n", c.groupby(["ticker", "method"], sort=False).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
dr = pd.DataFrame(drivers); dr.to_csv(DIAG / "F_drivers.csv", index=False)
print("\nF. LINEAR DRIVERS of target weight (scaled features). R2 + mean coef + seeds agreeing with sign of mean")
m = dr.groupby(["ticker", "method"], sort=False).mean(numeric_only=True).drop(columns="seed")
print(m.round(3).to_string())
agree = dr.groupby(["ticker", "method"], sort=False)[list(MARKET_FEATURES)].agg(lambda s: max((s > 0).sum(), (s < 0).sum()))
print(agree.to_string())
print("\nFee SNR: observations to detect a 50% rebalance fee (5 bps) at 2 sigma:")
for t in TICKERS:
    s = frames[t].loc[SPLITS["train"][0]:SPLITS["train"][1], "Close"].pct_change().std()
    print(f" {t}: train daily std {s:.4f} -> n = {(2 * s / 0.0005) ** 2:,.0f}; train days 1762; passes over Train at 204,800 steps = {204800 / 1762:.0f}")
